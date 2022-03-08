# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import random
from copy import copy
import tarfile
from tqdm import tqdm
import os, json
from .annotate import create_annotations
from ..monolingual_sql import MonolingualSQL
import torch

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['==', '>=', '<=']
syms = ['SELECT', 'WHERE', '&']
reserved_tokens_sql = syms + agg_ops[1:] + cond_ops + ['(', ')', '[CLS]', '[SEP]', '[PAD]']
crappy_lines_conala = [98588, 93783]
lines_with_special_char = [12537, 15373, 45209, 45211, 45213, 51045, 51047, 52704, 52758, 52759, 56181, 56184]
special_chars = ['￥', '€'] + ['£']*10
replacement = 'val'


class Wikisql(MonolingualSQL):
    def __init__(self, name, split, tokenizer, args, monolingual=False):
        self.threshold = {'train': 200,
                          'dev': 512,
                          'test': 512}
        super(Wikisql, self).__init__('wikisql', split, tokenizer, args, monolingual)

    @staticmethod
    def input_types(question, header, types, tokenizer):
        types = list(map(lambda x: 1 if x == 'text' else 2, types))
        types.insert(0, 0)
        input_tokens = tokenizer(question, header)['input_ids']
        token_types = []
        j = 0
        for token in input_tokens:
            token_types.append(types[j])
            if token == 102:
                j += 1
        return ', '.join(list(map(str, token_types)))

    @staticmethod
    def create_label(query, question, header, tokenizer):
        full_tokens_str = ' '.join(reserved_tokens_sql + question.split() + header)
        full_tokens = sorted(list(set(tokenizer.encode(full_tokens_str, add_special_tokens=False))))
        if query is not None:
            query_tokens = tokenizer.encode(query)
        else:
            query_tokens = tokenizer([(question, ' [SEP] '.join(header))])
            query_tokens = query_tokens['input_ids'][0]

        lables = []
        for i, token in enumerate(query_tokens):
            lables.append(full_tokens.index(token))
        return lables, full_tokens, tokenizer.decode(query_tokens[1:-1])

    @staticmethod
    def query2str(query, header):
        agg_str = header[query['sel']]
        agg_op = agg_ops[query['agg']]
        if agg_op:
            agg_str = '{}({})'.format(agg_op, agg_str)
        where_str = ' && '.join(['{} {} {}'.format(header[i], cond_ops[o], ' '.join(v['words'])) for i, o, v in query['conds']])

        if len(query['conds']) > 0:
            return 'SELECT {} WHERE {}'.format(agg_str, where_str)
        else:
            return 'SELECT {}'.format(agg_str)

    @staticmethod
    def augment(query, header):
        query = query.split()
        sel_col = random.randint(0, len(header) - 1)
        new_column_name = header[sel_col]
        new_query = copy(query)
        column_index = 1
        if query[column_index].upper() in agg_ops:
            column_index += 2
        if query[column_index].lower().strip() != new_column_name.strip():
            new_query[column_index] = new_column_name
            return ' '.join(new_query)
        else:
            return new_query

    def _download_dataset(self):
        train_file_name = 'data/train.jsonl'
        if not os.path.exists(os.path.join(self.dir_name, train_file_name)):
            self._download_file(url='https://raw.githubusercontent.com/salesforce/WikiSQL/master/data.tar.bz2', file_name=train_file_name)
            tar = tarfile.open(os.path.join(self.dir_name, "data.tar.bz2"))
            tar.extractall(path=self.dir_name)
            create_annotations(os.path.join(self.dir_name, 'data'), self.dir_name)

    def _preprocess(self):
        json_file = os.path.join(self.dir_name, '{}.json'.format(self.split))
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                return json.load(f)
        dataset_lines = open(os.path.join(self.dir_name, '{}_annotated.jsonl'.format(self.split)), 'r').readlines()
        full_dataset = []
        print("Preprocessing {} ...".format(self.split))
        for index, data in enumerate(tqdm(dataset_lines)):
            data = json.loads(data)
            question = ' '.join(data['question']['words'])
            header = [' '.join(header['words']) for header in data['table']['header']]
            query = Wikisql.query2str(data['query'], header)
            header_string = ' [SEP] '.join(header)
            input_types = Wikisql.input_types(question, header_string, data['table']['type'], self.tokenizer)
            label, choices, query = Wikisql.create_label(query, question, header, self.tokenizer)
            source_labels, _, _ = Wikisql.create_label(None, question, header, self.tokenizer)
            example = {'intent': question,
                       'header': header_string,
                       'input_types': input_types,
                       'source_label': source_labels,
                       'snippet': query,
                       'choices': choices,
                       'label': label}
            full_dataset.append(example)
        with open(json_file, 'w') as f:
            json.dump(full_dataset, f)
        return full_dataset

    def tokenize(self, tokenizer, json_object):
        intent = tokenizer(json_object['intent'], json_object['header'],
                           max_length=self.threshold['test'], padding=False, truncation=True)
        target = tokenizer(json_object['snippet'], max_length=self.threshold['test'], padding=False, truncation=True)
        if not self.args.pointer_network:
            return {'intent': intent, 'snippet': target}
        else:
            source_label = json_object['source_label']
            intent['source_label'] = source_label
            choice_dict = {'input_ids': json_object['choices'],
                           'attention_mask': torch.ones(len(json_object['choices'])),
                           'token_type_ids': torch.zeros(len(json_object['choices']))}
            return {'intent': intent, 'snippet': target,
                    'choices': choice_dict, 'label': json_object['label']}



