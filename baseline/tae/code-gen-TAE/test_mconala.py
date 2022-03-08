# Run evaluation on Multilingual CoNaLa dataset (trans-test)

from torch.utils.data import DataLoader
import argparse
import torch
from model import Model
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle
from booster.utils import EMA
from evaluation.compute_eval_metrics import compute_metric
import os
from evaluation.evaluation import generate_hypothesis
from utils import generate_model_name
from torch.nn.utils.rnn import pad_sequence

from dataset_preprocessing.conala import Conala
from dataset_preprocessing.multilingual_conala import (
    MConalaFlores101, MConalaMarianMT, MConalaM2M, 
)


dataset_classes = {
    'conala': Conala, 
    'es-101': MConalaFlores101, 
    'ja-101': MConalaFlores101, 
    'ru-101': MConalaFlores101, 
    'es-mmt': MConalaMarianMT, 
    'ja-mmt': MConalaMarianMT, 
    'ru-mmt': MConalaMarianMT, 
    'es-m2m': MConalaM2M, 
    'ja-m2m': MConalaM2M, 
    'ru-m2m': MConalaM2M
}


def load_dataset(args, tokenizer):
    splits = ['test']
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


def test_mconala(args):
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
    model.to(args.device)
        
    test_dataset = load_dataset(args, model.tokenizer)
    model_name = generate_model_name(args)
    print("model name", model_name)
    writer = SummaryWriter(log_dir=args.save_dir+'/logs/{}/'.format(model_name) + model_name[:-4])

    print('test set size', len(test_dataset))
    print("example of parallel data")

    # do evaluation 
    with torch.no_grad():
        model.load_state_dict(torch.load(os.path.join(args.save_dir, model_name)))
        model.eval()
        
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=preprocess_batch)

        loader = {'test': test_loader}
        for split in ['test']:
            for search in ['greedy', 'beam']:
                file = os.path.join(args.save_dir, '{}_hype_{}_{}.pt'.format(args.dataset_name, split, search))
                if os.path.exists(file):
                    generated_set = pickle.load(open(file, 'rb'))
                else:
                    generated_set = generate_hypothesis(args, loader[split], model, search=search)
                    print(f"generated_set: {len(generated_set)}")
                    with open(file, 'wb') as f:
                        pickle.dump(generated_set, f)
                print(f"args.dataset_name: {args.dataset_name}")
                print(f"generated_set: {len(generated_set)}")
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
    parser.add_argument('--dataset_name', type=str, default='conala')
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

    parser.add_argument('--use_conala_model', action='store_true', 
        help='To directly load the model pre-trained on English CoNala. ')
    parser.add_argument('--use_mconala_model', action='store_true', 
        help='To reproduce the result in Multilingual CoNaLa translate-test setting. ')
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    args = parser.parse_args()
    if args.dataset_name == 'conala':
        args.python = True
    elif args.dataset_name in ['es-mmt', 'ja-mmt', 'ru-mmt']: 
        args.python = True 
    elif args.dataset_name in ['es-101', 'ja-101', 'ru-101']: 
        args.python = True 
    elif args.dataset_name in ['es-m2m', 'ja-m2m', 'ru-m2m']: 
        args.python = True 
    else:
        raise Exception("Wrong Dataset Name!")
    
    args.no_encoder = False
    if args.language_model:
        args.no_encoder = True

    if args.generate_back_translation:
        args.translate_backward = True

    print(args)
    test_mconala(args)
