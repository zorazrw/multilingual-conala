# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import torch
from model import Model
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle
from booster.utils import EMA
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from evaluation.evaluation import evaluate
from evaluation.compute_eval_metrics import compute_metric
from utils import compute_loss, get_next_batch
import os
from evaluation.evaluation import generate_hypothesis
from utils import generate_model_name
from torch.nn.utils.rnn import pad_sequence
from dataset_preprocessing.django import Django
from dataset_preprocessing.conala import Conala
from dataset_preprocessing.small_sql import SmallSQL
from dataset_preprocessing.wikisql.wikisql import Wikisql


dataset_classes = {'django': Django,
                   'conala': Conala,
                   'atis': SmallSQL,
                   'geography': SmallSQL,
                   'wikisql': Wikisql}


def load_dataset(args, tokenizer):
    splits = ['train', 'dev', 'test']
    datasets = []
    for split in splits:
        dataset = dataset_classes[args.dataset_name](args.dataset_name, split, tokenizer, args)
        datasets.append(dataset)
    return (*datasets,) if len(datasets) > 1 else dataset


def preprocess_batch(data):
    data_intents = [d['intent'] for d in data]
    data_snippets = [d['snippet'] for d in data]
    keys = ['input_ids', 'attention_mask', 'token_type_ids']
    source_dict = {key: pad_sequence([torch.tensor(d[key]) for d in data_intents], batch_first=True, padding_value=0)
                              for key in keys}
    target_dict = {key: pad_sequence([torch.tensor(d[key]) for d in data_snippets], batch_first=True, padding_value=0)
                                for key in keys}
    extra_info = {}
    if args.pointer_network:
        source_dict['source_label'] = pad_sequence([torch.tensor(d['source_label']) for d in data_intents],
                                                   batch_first=True, padding_value=0)
        data_choices = [d['choices'] for d in data]
        extra_info['choices'] = {key: pad_sequence([torch.tensor(d[key]) for d in data_choices], batch_first=True, padding_value=0)
                              for key in keys}
        extra_info['label'] = pad_sequence([torch.tensor(d['label']) for d in data], batch_first=True, padding_value=0)
    return {'source': source_dict, 'target': target_dict, **extra_info}


def print_dataset_length_info(train_dataset):
    length = []
    for i in range(len(train_dataset)):
        length.append(len(train_dataset[i]['intent']['input_ids']))
    print("min", min(length))
    print("max", max(length))
    length = np.array(length)
    print('std', np.std(length))
    print('mean', np.mean(length))


def train(args):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pretrained_weights = "bert-base-uncased"
    model = Model(pretrained_weights, args)
    params_except_encoder = []
    for name, p in model.named_parameters():
        if not name.startswith('encoder.'):
            params_except_encoder.append(p)

    decoder_optimizer = torch.optim.Adam(lr=args.decoder_lr, params=params_except_encoder)
    encoder_optimizer = torch.optim.Adam(lr=args.encoder_lr, params=model.encoder.parameters())
    encoder_scheduler = ExponentialLR(encoder_optimizer, gamma=1)
    decoder_scheduler = LambdaLR(decoder_optimizer, lr_lambda=lambda step: (step+1)/args.warmup_steps if step<args.warmup_steps else args.lr_decay**(step-args.warmup_steps))
    model.to(args.device)
    if args.EMA:
        ema_model = EMA(model, args.ema_param)
    train_dataset, valid_dataset, test_dataset = load_dataset(args, model.tokenizer)
    print_dataset_length_info(train_dataset)
    if args.small_dataset:
        train_dataset = train_dataset[:round(len(train_dataset)*args.percentage/100)]
    else:
        args.percentage = 100
    if args.copy_bt:
        args.batch_size = int(args.batch_size//(1+args.monolingual_ratio))
    print("Effective batch size", args.batch_size)
    model_name = generate_model_name(args)
    print("model name", model_name)
    writer = SummaryWriter(log_dir=args.save_dir+'/logs/{}/'.format(model_name) + model_name[:-4])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=preprocess_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=preprocess_batch)
    if args.copy_bt:
        copy_dataset = dataset_classes[args.dataset_name](name=args.dataset_name, split='train', tokenizer=model.tokenizer, args=args, monolingual=True)
        copy_loader = DataLoader(copy_dataset, batch_size=int(args.batch_size * args.monolingual_ratio),
                                 shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=preprocess_batch)
        copy_iter = iter(copy_loader)
        print('copy dataset size', len(copy_dataset))

    print('train set size:', len(train_dataset))
    print('dev set size', len(valid_dataset))
    print('test set size', len(test_dataset))
    print("example of parallel data")
    print(model.tokenizer.decode(train_dataset[0]['intent']['input_ids']))
    print(model.tokenizer.decode(train_dataset[0]['snippet']['input_ids']))
    if args.copy_bt:
        print(len(copy_dataset))
        print("example of monolingual data")
        print(model.tokenizer.decode(copy_dataset[0]['intent']['input_ids']))
        print(model.tokenizer.decode(copy_dataset[0]['snippet']['input_ids']))

    resume_file = os.path.join(args.save_dir, 'resume.pth')
    if not args.just_evaluate:
        if os.path.exists(resume_file):
            print("resume is loaded")
            checkpoint = torch.load(resume_file)
            model.load_state_dict(checkpoint['model_to_evaluate'])
            ema_model = EMA(model, args.ema_param)
            model.load_state_dict(checkpoint['model_to_train'])
            if not args.no_encoder:
                encoder_optimizer.load_state_dict(checkpoint['enc_optimizer_state'])
            decoder_optimizer.load_state_dict(checkpoint['dec_optimizer_state'])
            best_criteria = checkpoint['best_criteria']
            begin_epoch = checkpoint['epoch']
            early_stopping = checkpoint['early_stopping']
        else:
            best_criteria = -float('inf')
            begin_epoch = 0
            early_stopping = 0

        for epoch in range(begin_epoch, args.epochs):
            averaged_loss = 0
            print('Epoch :', epoch + 1, "Early Stopping:", early_stopping,
                  "encoder lr: ", encoder_scheduler.get_lr() if not args.no_encoder else "no encoder",
                  "decoder_lr", decoder_scheduler.get_lr())
            model.train()
            for data in tqdm(train_loader):
                loss, logits, choices = compute_loss(args, data, model)
                loss = loss.sum(1)
                if args.copy_bt:
                    copy_data = None
                    if args.copy_bt:
                        copy_data, copy_iter = get_next_batch(iterator=copy_iter, loader=copy_loader)
                        copy_data['source'] = copy_data['target']
                    loss_bt, _, _ = compute_loss(args, copy_data, model, no_context_update=args.no_encoder_update_for_bt)
                    loss = torch.cat([loss, loss_bt.sum(1)], dim=0)
                    if args.copy_bt:
                        del copy_data
                loss = loss.mean()
                averaged_loss += loss.item()*len(data['source']['input_ids'])
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                if args.EMA:
                    ema_model.update()
                decoder_scheduler.step()
            encoder_scheduler.step()
            averaged_loss = averaged_loss / len(train_loader.dataset)
            writer.add_scalar('Loss/train', averaged_loss, epoch)
            with torch.no_grad():
                model_to_evaluate = ema_model.model if args.EMA else model
                validation_loss = evaluate(args, valid_loader, model_to_evaluate, split='dev')
                print('validation loss', validation_loss)
                if (epoch + 1) % args.eval_interval == 0:
                    greedy_hype = generate_hypothesis(args, valid_loader, model_to_evaluate, search='greedy')
                    metrics = compute_metric(greedy_hype, args.dataset_name, split='dev', tokenizer=model.tokenizer, args=args)
                    writer.add_scalar('Loss/dev', validation_loss, epoch)
                    print('exact match accuracy', metrics['exact_match'])
                    print('bleu:', metrics['bleu'])
                    criteria = metrics['exec_acc'] if args.dataset_name == 'wikisql' \
                        else metrics['bleu'] if (args.dataset_name == 'conala' or args.dataset_name == 'magic')\
                        else metrics['exact_match']
                    print("criteria", criteria)
                    writer.add_scalar('evaluation metric', criteria, epoch)
                    if args.early_stopping:
                        if best_criteria < criteria:
                            best_criteria = criteria
                            torch.save(model_to_evaluate.state_dict(), os.path.join(args.save_dir, model_name))
                            early_stopping = 0
                        else:
                            early_stopping += 1
                        if early_stopping >= args.early_stopping_epochs:
                            break
                    else:
                        torch.save(model_to_evaluate.state_dict(), os.path.join(args.save_dir, model_name))
                    print("resume.pth is saved")
                    torch.save({
                        'epoch': epoch+1,
                        'model_to_evaluate': model_to_evaluate.state_dict(),
                        'model_to_train': model.state_dict(),
                        'enc_optimizer_state': encoder_optimizer.state_dict() if not args.no_encoder else None,
                        'dec_optimizer_state': decoder_optimizer.state_dict(),
                        'best_criteria': best_criteria,
                        'early_stopping': early_stopping
                    }, resume_file)


    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(args.save_dir, model_name)))
        model.eval()
        valid_loader = DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=preprocess_batch)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=preprocess_batch)

        loader = {'train': train_loader, 'dev': valid_loader, 'test': test_loader}
        for split in ['dev', 'test']:
            for search in ['greedy', 'beam']:
                file = os.path.join(args.save_dir, 'hype_{}_{}.pt'.format(split, search))
                if os.path.exists(file):
                    generated_set = pickle.load(open(file, 'rb'))
                else:
                    generated_set = generate_hypothesis(args, loader[split], model, search=search)
                    with open(file, 'wb') as f:
                        pickle.dump(generated_set, f)
                metrics = compute_metric(generated_set, args.dataset_name, split=split, tokenizer=model.tokenizer, args=args)
                print('{} {} accuracy'.format(split, search), metrics['exact_match'])
                if search == 'beam':
                    print('{} {} oracle accuracy'.format(split, search), metrics['exact_oracle_match'])
                print('{} {} bleu score'.format(split, search), metrics['bleu'])
                print("{} {} exececution accuracy".format(split, search), metrics['exec_acc'])

    writer.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--encoder_lr', type=float, default=1e-5)
    parser.add_argument('--decoder_lr', type=float, default=7.5e-5)
    parser.add_argument('--lambd', type=float, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--decoder_layers', type=int, default=4)
    parser.add_argument('--dataset_name', type=str, default='wikisql')
    parser.add_argument('--save_dir', type=str, default='/home/sajad/pretrain_sp_decocder')
    parser.add_argument('--just_evaluate', action='store_true', default=False)
    parser.add_argument('--just_initialize', action='store_true', default=False)
    parser.add_argument('--auxilary_lm', action='store_true', default=False)
    parser.add_argument('--valid_batch_size', type=int, default=50)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--beam_num', type=int, default=10)
    parser.add_argument('--beam_search_base', type=int, default=3)
    parser.add_argument('--beam_search_alpha', type=float, default=0.9)
    parser.add_argument('--extra_encoder_layers', type=int, default=1)
    parser.add_argument('--early_stopping_epochs', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--percentage', type=int, default=10)
    parser.add_argument('--language_model', action='store_true', default=False)
    parser.add_argument('--use_authentic_data', action='store_true', default=False)
    parser.add_argument('--use_tagged_back', action='store_true', default=False)
    parser.add_argument('--extra_encoder', action='store_true', default=False)
    parser.add_argument('--small_dataset', action='store_true', default=False)
    parser.add_argument('--combined_eval', action='store_true', default=False)
    parser.add_argument('--use_real_source', action='store_true', default=False)
    parser.add_argument('--combined_training', action='store_true', default=False)
    parser.add_argument('--create_mapping', action='store_true', default=False)
    parser.add_argument('--pointer_network', action='store_true', default=False)
    parser.add_argument('--gating', action='store_true', default=False)
    parser.add_argument('--extra_linear', action='store_true', default=True)
    parser.add_argument('--extra_copy_attention_linear', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--monolingual_ratio', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--temp', type=float, default=1.)
    parser.add_argument('--mono_min_prob', type=float, default=.1)
    parser.add_argument('--label_smoothing', type=float, default=.1)
    parser.add_argument('--translate_backward', action='store_true', default=False)
    parser.add_argument('--copy_bt', action='store_true', default=False)
    parser.add_argument('--add_noise', action='store_true', default=False)
    parser.add_argument('--use_back_translation', action='store_true', default=False)
    parser.add_argument('--generate_back_translation', action='store_true', default=False)
    parser.add_argument('--no_encoder_update_for_bt', action='store_true', default=False)
    parser.add_argument('--just_analysis', action='store_true', default=False)
    parser.add_argument('--early_stopping', action='store_true', default=False)
    parser.add_argument('--use_copy_attention', action='store_true', default=True)
    parser.add_argument('--dummy_source', action='store_true', default=False)
    parser.add_argument('--dummy_question', action='store_true', default=False)
    parser.add_argument('--python', action='store_true', default=False)
    parser.add_argument('--EMA', action='store_true', default=True)
    parser.add_argument('--random_encoder', action='store_true', default=False)
    parser.add_argument('--sql_augmentation', action='store_true', default=False)
    parser.add_argument('--sql_where_augmentation', action='store_true', default=False)
    parser.add_argument('--use_column_type', action='store_true', default=False)
    parser.add_argument('--use_codebert', action='store_true', default=False)
    parser.add_argument('--fixed_copy', action='store_true', default=False)
    parser.add_argument('--combine_copy_with_real', action='store_true', default=False)
    parser.add_argument('--no_schema', action='store_true', default=False)
    parser.add_argument('--no_linear_opt', action='store_true', default=False)
    parser.add_argument('--fix_linear_layer', action='store_true', default=False)
    parser.add_argument('--use_gelu', action='store_true', default=True)
    parser.add_argument('--ema_param', type=float, default=.999)
    parser.add_argument('--bleu_threshold', type=float, default=0)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--n', type=int, default=256)
    parser.add_argument('--warmup_steps', type=int, default=0)

    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    args = parser.parse_args()
    if args.dataset_name == 'django' or args.dataset_name == 'conala':
        args.python = True
    elif args.dataset_name == 'wikisql':
        if not args.translate_backward:
            args.pointer_network = True
        args.beam_num = 5
        args.test_batch_size = 100
        args.valid_batch_size = 100
        args.eval_interval = 5
        args.beam_search_base = 0
        args.beam_search_alpha = 1
        args.early_stopping_epochs = args.early_stopping_epochs//args.eval_interval+1
        if args.small_dataset is False:
            args.epochs = 10
        else:
            args.epochs = 100

    elif args.dataset_name =='magic':
        args.eval_interval = 5
    elif args.dataset_name in ['atis', 'geo', 'imdb', 'scholar', 'advising', 'academic']:
        args.beam_search_base = 0
        args.beam_search_alpha = 1
    else:
        raise Exception("Wrong Dataset Name!")
    args.no_encoder = False
    if args.language_model:
        args.no_encoder = True

    if args.generate_back_translation:
        args.translate_backward = True

    print(args)
    train(args)
