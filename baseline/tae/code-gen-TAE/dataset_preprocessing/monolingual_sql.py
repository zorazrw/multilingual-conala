# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from tqdm import tqdm
import pickle
from pglast import Node, parse_sql
import six
import abc
import os
from .dataset import Dataset

@six.add_metaclass(abc.ABCMeta)
class MonolingualSQL(Dataset):
    def __init__(self, name, split, tokenizer, args, monolingual=False):
        super(MonolingualSQL, self).__init__(name, split, tokenizer, args, monolingual)

    def _download_monolingual(self):
        if not os.path.exists(os.path.join(self.dir_name, self.monolingual_file)):
            self._download_file(
                'https://raw.githubusercontent.com/LittleYUYU/StackOverflow-Question-Code-Dataset/master/annotation_tool/data/code_solution_labeled_data/source/sql_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle',
                self.monolingual_file)
            os.rename(os.path.join(self.dir_name,
                                   'sql_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle'),
                      os.path.join(self.dir_name, 'mono_single_sql.pickle'))
            self._download_file(
                'https://raw.githubusercontent.com/LittleYUYU/StackOverflow-Question-Code-Dataset/master/annotation_tool/data/code_solution_labeled_data/source/sql_how_to_do_it_by_classifier_multiple_iid_to_code.pickle',
                self.monolingual_file)
            os.rename(os.path.join(self.dir_name, 'sql_how_to_do_it_by_classifier_multiple_iid_to_code.pickle'),
                      os.path.join(self.dir_name, 'mono_multi_sql.pickle'))

    def _preprocess_monolingual(self, args):
        with open(os.path.join(self.dir_name, 'mono_single_sql.pickle'), 'rb') as f:
            monolingual_sql_single = pickle.load(f)
        with open(os.path.join(self.dir_name, 'mono_multi_sql.pickle'), 'rb') as f:
            monolingual_sql_multi = pickle.load(f)
        monolingual_sql = {**monolingual_sql_single, **monolingual_sql_multi}
        json_file = os.path.join(self.dir_name, self.monolingual_file)
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                return json.load(f)
        counter = 0
        dataset = []
        length_threshold = 12 if args.dataset_name == 'geo' else 19 if args.dataset_name=='atis' else 80
        print("lengh threshold", length_threshold)
        for index, (id, query) in enumerate(tqdm(monolingual_sql.items())):
            query = query.lower().strip()
            if len(query) > 6 and 'select' in query:
                if ('--') in query:
                    continue
                try:
                    Node(parse_sql(query))  # just to make sure the query has a valid grammar
                except:
                    continue
                query = query.replace('\n', ' ').replace('"', '').replace('\'', '').replace(';',
                                                                                            '')  # simplify the query
                if len(query.split()) > length_threshold or len(query) < 8 or query[:6] != 'select':
                    continue
                query = ' '.join(query.lower().strip().split())
                example = {'intent': query,
                           'snippet': query}
                dataset.append(example)
                counter += 1
        with open(json_file, 'w') as f:
            json.dump(dataset, f)
        return dataset