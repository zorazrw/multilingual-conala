# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
import pickle
import numpy as np
from dataset_preprocessing.wikisql.lib.query import Query
from dataset_preprocessing.wikisql.lib.db_engine import DBEngine
from utils import detokenize_query
from evaluation.compute_bleu import compute_bleu


def is_equal(translation_token, tokenized_source):
    if (len(tokenized_source)-1) <= len(translation_token):
        correct_tokens = ((translation_token[:len(tokenized_source) - 1] == tokenized_source[1:]).float()).sum()
        return correct_tokens == (len(tokenized_source) - 1)
    else:
        return False


def compute_metric(translation_corpus, dataset_name, split, tokenizer=None, section=None, args=None):
    dataset = os.path.join('data/{}/{}.json'.format(dataset_name, split))
    with open(dataset) as dataset_file:
        dataset_object = json.loads(dataset_file.read())
    with open('data/{}/{}_order.json'.format(dataset_name, split), 'rb') as f:
        indices = pickle.load(f)
    dataset_object = np.array(dataset_object)[indices].tolist()
    if dataset_name == 'wikisql':
        annotated_dataset_file = os.path.join('data/{}/{}_annotated.jsonl'.format(dataset_name, split))
        annotated_dataset_object = open(annotated_dataset_file, 'r').readlines()
        annotated_dataset_object = np.array(annotated_dataset_object)[indices].tolist()
        dbengine = DBEngine('data/wikisql/{}.db'.format(split))

    exact_match_acc = 0
    oracle_exact_match_acc = 0
    execution_acc = 0

    if section is None:
        section = range(len(dataset_object))
    mistakes = []
    for index in section:
        translation = translation_corpus[index]
        reference = dataset_object[index]['snippet'].lower()
        if args.dataset_name == 'wikisql':
            annotated_data = json.loads(annotated_dataset_object[index])
        tokenized_source = tokenizer.encode(reference, padding=True, truncation=True, return_tensors="pt")[0]
        if isinstance(translation, dict):
            if is_equal(translation['token'], tokenized_source.to('cuda')):
                exact_match_acc += 1
            else:
                mistakes.append((index, translation['str']))
            translation = translation['str']
            translation_corpus[index] = translation
        else:
            if is_equal(translation[0]['token'], tokenized_source):
                exact_match_acc += 1
            else:
                mistakes.append((index, translation[0]['str']))
            for trans in translation:
                if is_equal(trans['token'], tokenized_source):
                    oracle_exact_match_acc += 1
                    break
            translation = translation[0]['str']
            translation_corpus[index] = translation

        if dataset_name == 'wikisql':
            header = dataset_object[index]['header'].split('[SEP]')
            reference_query = Query.from_tokenized_dict(annotated_data['query'])
            result2 = dbengine.execute_query(annotated_data['table_id'], reference_query, lower=True)
            try:
                predicted_query = Query.from_real_sequence(translation, header)
                if predicted_query is None:
                    continue
                detokenized_prediction = detokenize_query(predicted_query,
                                                          annotated_data['question'],
                                                          annotated_data['table']['type'])
                result1 = dbengine.execute_query(annotated_data['table_id'], detokenized_prediction,
                                                 lower=True)
                if result1 == result2:
                    execution_acc += int(result1 == result2)
            except Exception as e:
                continue

    bleu, bleu_sentence = compute_bleu(translation_corpus, dataset_object, section, args=args)
    metrics = {'bleu': bleu,
               'bleu_sentence': bleu_sentence,
               'exact_match': exact_match_acc/len(section),
               'exact_oracle_match': oracle_exact_match_acc/len(section),
               'exec_acc': execution_acc/len(section),
               'mistakes': mistakes
               }
    return metrics


