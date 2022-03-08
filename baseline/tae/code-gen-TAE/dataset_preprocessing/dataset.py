# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import six
import abc
import os
import wget
from torch.utils.data import Dataset
from functools import partial
import pickle
import numpy as np
from tqdm import tqdm

@six.add_metaclass(abc.ABCMeta)
class Dataset(Dataset):

    def __init__(self, name, split, tokenizer, args, monolingual=False):
        self.name = name
        self.dir_name = os.path.join('data', name)
        self.monolingual = monolingual
        self.monolingual_file = 'train_monolingual.jsonl'
        if not os.path.exists('data'):
            os.mkdir('data')
        if monolingual:
            self.split = 'train'
        else:
            self.split = split
        self.tokenizer = tokenizer
        self.args = args
        binary_file = os.path.join(self.dir_name, '{}{}.bin'.format(split, '_mono' if monolingual else ''))
        if os.path.exists(binary_file):
            self.data = pickle.load(open(binary_file, 'rb'))
        else:
            if not os.path.exists(self.dir_name):
                os.mkdir(self.dir_name)
            if monolingual:
                self.data = self.preprocess_monolingual(args)
            else:
                self.data = self.preprocess()
            filter_function = partial(self.tokenize, tokenizer)
            print('tokenizing the {} data....'.format(self.split))
            self.data = list(map(filter_function, tqdm(self.data)))
            print('tokenization is done.')
            if self.split == 'train':
                self.data = list(filter(lambda x: len(x['intent']['input_ids']) < self.threshold[self.split] and
                                                  len(x['snippet']['input_ids']) < self.threshold[self.split], self.data))
            else:
                indices = sorted(range(len(self.data)), key=lambda item: len(self.data[item]['snippet']['input_ids']))
                with open(os.path.join(self.dir_name, '{}_order.json'.format(split)), 'wb') as f:
                    pickle.dump(indices, f)
                self.data = np.array(self.data)[indices].tolist()
            pickle.dump(self.data, open(binary_file, 'wb'), protocol=4)

    @abc.abstractmethod
    def _download_dataset(self):
        pass

    @abc.abstractmethod
    def _preprocess(self):
        pass

    def preprocess(self):
        self._download_dataset()
        return self._preprocess()

    @abc.abstractmethod
    def _preprocess_monolingual(self, args):
        pass

    @abc.abstractmethod
    def _download_monolingual(self):
        pass

    def preprocess_monolingual(self, args):
        self._download_monolingual()
        return self._preprocess_monolingual(args)

    def _download_file(self, url, file_name):
        self.file_name = os.path.join(self.dir_name, file_name)
        if not os.path.exists(self.file_name):
            wget.download(url, out=self.dir_name)

    def tokenize(self, tokenizer, json_object):
        if isinstance(json_object['intent'], dict):
            intent = ' '.join(json_object['intent']['words'])
        else:
            intent = json_object['intent']
        return {'intent': tokenizer(intent, max_length=self.threshold['test'], padding=False, truncation=True),
                'snippet': tokenizer(json_object['snippet'], max_length=self.threshold['test'], padding=False, truncation=True)}

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

