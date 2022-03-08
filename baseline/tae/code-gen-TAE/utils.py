# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import torch
from babel.numbers import parse_decimal, NumberFormatError
from dataset_preprocessing.wikisql.lib.query import Query
import re
import unicodedata

num_re = re.compile(r'[-+]?\d*\.\d+|\d+')


def strip_accents(text):
    return ''.join(char for char in
                   unicodedata.normalize('NFKD', text)
                   if unicodedata.category(char) != 'Mn')


def my_annotate(sentence):
    gloss = []
    tokens = []
    after = []
    punctuation = {'.', ',', "'", '"', '/', '\\', '&', '*', '(', ')', '%', '$', '€', '£', '￥', '￥', '’', '–', '·', '—',
                   '-', '#', '!', '?', '+', '^', '=', ':', ';', '{', '}', '[', ']', '_'}
    word = ''
    for ind in range(len(sentence)):
        s = sentence[ind]
        if s == ' ':
            if len(word) > 0:
                gloss.append(word)
                after.append(' ')
                tokens.append(strip_accents(word.lower()))
                word = ''
            else:
                continue
        elif s in punctuation:
            if len(word)>0:
                gloss.append(word)
                after.append('')
                tokens.append(strip_accents(word.lower()))
                word = ''
            tokens.append(s)
            gloss.append(s)
            if ind < (len(sentence)-1) and sentence[ind+1] == ' ':
                after.append(' ')
            else:
                after.append('')
        else:
            word += s
    if len(word)>0:
        gloss.append(word)
        after.append('')
        tokens.append(strip_accents(word.lower()))
    return {'gloss': gloss, 'words': tokens, 'after': after}


def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def LabelSmoothingCrossEntropy(preds, target, args, choices_attention=None):
    log_preds = F.log_softmax(preds, dim=1)
    nll = F.nll_loss(log_preds, target, reduction='none')
    if args.pointer_network and choices_attention is not None:
        coeff = (1/choices_attention.sum(1).float()).unsqueeze(1)
        masked_log_preds = log_preds.masked_fill(torch.isinf(log_preds), value=0.0)
        loss = -masked_log_preds.sum(dim=1)
        return linear_combination(loss * coeff, nll, args.label_smoothing)
    elif not args.pointer_network:
        loss = -log_preds.sum(dim=1) / preds.size()[1]
        return linear_combination(loss, nll, args.label_smoothing)
    else:
        return nll


def compute_loss(args, data, model, target_input=None, no_context_update=False, encoder_output_saved=None):
    target_input_model = None
    if target_input is not None:
        target_input_model = target_input
    logits, target, choices, labels, hidden = model(data, target_input=target_input_model,
                                                     no_context_update=no_context_update,
                                                    encoder_output_saved=encoder_output_saved)
    if args.pointer_network:
        labels = labels[:, 1:target['input_ids'].shape[1]].to(args.device)
    else:
        labels = target['input_ids'][:, 1:]
    if target_input is not None:
        loss = None
    else:
        loss = LabelSmoothingCrossEntropy(torch.transpose(logits, 1, 2), labels, args,
                         choices['attention_mask'] if args.pointer_network else None)
        loss = (loss*target['attention_mask'][:, 1:])
    return loss, logits, choices


def generate_model_name(args):
    model_first_token = args.dataset_name
    if args.use_conala_model: model_first_token = "conala"
    if args.use_mconala_model: model_first_token = "mconala"
    extention = '_LM' if args.language_model is True else ''
    if extention == '_LM':
        if args.python:
            model_first_token = 'python'
        elif args.dataset_name == 'magic':
            model_first_token = 'java'
        else:
            model_first_token = 'sql'

    model_name = '{}_model{}{}_combined_training={}_seed={}{}{}{}{}.pth'.format(
        model_first_token,
        extention,
        str(args.percentage) if args.small_dataset is True else '',
        args.combined_training,
        args.seed,
        '_beta=' + str(args.beta) if args.combined_training else '',
        '_tmp=' + str(args.temp) if args.combined_training else '',
        '_trns_back=' + str(args.translate_backward),
        '_use_backtr=' + str(args.use_back_translation) +
        '_lmd=' + str(args.lambd) +
        '_cp_bt=' + str(args.copy_bt) +
        '_add_no=' + str(args.add_noise) +
        '_no_en_upd=' + str(args.no_encoder_update_for_bt) +
        '_ratio=' + str(args.monolingual_ratio) +
        '_ext_li=' + str(args.extra_linear) +
        '_ext_cp_li=' + str(args.extra_copy_attention_linear) +
        '_cp_att=' + str(args.use_copy_attention) +
        '_EMA=' + str(args.EMA)[0] +
        '_rnd_enc=' + str(args.random_encoder)[0] +
        '_de_lr=' + str(args.decoder_lr) +
        '_mmp=' + str(args.mono_min_prob) +
        '_saug=' + str(args.sql_augmentation)[0] +
        '_dums=' + str(args.dummy_source)[0] +
        '_dumQ=' + str(args.dummy_question)[0] +
        '_rsr=' + str(args.use_real_source)[0] +
        '_fc=' + str(args.fixed_copy)[0] +
        '_ccr=' + str(args.combine_copy_with_real)[0]
    )
    return model_name


def get_next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def find_sub_sequence(sequence, query_seq):
    for i in range(len(sequence)):
        if sequence[i: len(query_seq) + i] == query_seq:
            return i, len(query_seq) + i
    raise IndexError


def my_detokenize_code(code, dictionary):
    code = code.replace('.', ' . ').replace(',', ' , ').replace("'", " ' ")\
               .replace('!', ' ! ').replace('"', ' " ').split()
    literal = []
    intent = dictionary['words']
    i = 0
    while i < len(code):
        index_i = -1
        max_length = 1
        for j in range(len(intent)):
            if code[i] == intent[j]:
                length = 1
                while (i+length) < len(code) and (j+length) < len(intent) and code[i+length] == intent[j+length]:
                    length += 1
                if length > max_length:
                    max_length = length
                    index_i = j
        if index_i == -1:
            literal.append(code[i]+' ')
            i += 1
        else:
            i += max_length
            for j in range(max_length):
                literal.append(dictionary['gloss'][index_i+j]+dictionary['after'][index_i+j])
    return ''.join(literal)


def my_detokenize(tokens, token_dict, raise_error=False):
    literal = []
    try:
        start_idx, end_idx = find_sub_sequence(token_dict['words'], tokens)
        for idx in range(start_idx, end_idx):
            literal.extend([token_dict['gloss'][idx], token_dict['after'][idx]])

        val = ''.join(literal).strip()
    except IndexError:
        if raise_error:
            raise IndexError('cannot find the entry for [%s] in the token dict [%s]' % (' '.join(tokens),
                                                                                        ' '.join(token_dict['words'])))
        for token in tokens:
            match = False
            for word, gloss, after in zip(token_dict['words'], token_dict['gloss'], token_dict['after']):
                if token == word:
                    literal.extend([gloss, after])
                    match = True
                    break

            if not match and raise_error:
                raise IndexError('cannot find the entry for [%s] in the token dict [%s]' % (' '.join(tokens),
                                                                                            ' '.join(
                                                                                                token_dict['words'])))
            if not match:
                literal.extend(token)
        val = ''.join(literal).strip()
    return val


def detokenize_query(query, tokenized_question, table_header_type):
    detokenized_conds = []
    for i, (col, op, val) in enumerate(query.conditions):
        val_tokens = val.split(' ')
        detokenized_cond_val = my_detokenize(val_tokens, tokenized_question)

        if table_header_type[col] == 'real' and not isinstance(detokenized_cond_val, (int, float)):
            if ',' not in detokenized_cond_val:
                try:
                    detokenized_cond_val = float(parse_decimal(detokenized_cond_val))
                except NumberFormatError as e:
                    try:
                        detokenized_cond_val = float(num_re.findall(detokenized_cond_val)[0])
                    except: pass
        detokenized_conds.append((col, op, detokenized_cond_val))
    detokenized_query = Query(sel_index=query.sel_index, agg_index=query.agg_index, conditions=detokenized_conds)
    return detokenized_query
