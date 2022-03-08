# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from zipfile import ZipFile
import json
from tqdm import tqdm
import six
import abc
import os
from .dataset import Dataset


@six.add_metaclass(abc.ABCMeta)
class MonolingualPython(Dataset):
    def __init__(self, name, split, tokenizer, args, monolingual=False):
        super(MonolingualPython, self).__init__(name, split, tokenizer, args, monolingual)

    def _download_monolingual(self):
        if not os.path.exists(os.path.join(self.dir_name, self.monolingual_file)):
            self._download_file('http://www.phontron.com/download/conala-corpus-v1.1.zip', self.monolingual_file)
            with ZipFile(os.path.join(self.dir_name, 'conala-corpus-v1.1.zip'), 'r') as zipObj:
                zipObj.extract('conala-corpus/conala-mined.jsonl', self.dir_name)
                os.rename(os.path.join(self.dir_name, 'conala-corpus/conala-mined.jsonl'),
                          os.path.join(self.dir_name, self.monolingual_file))

    def _preprocess_monolingual(self, args):
        dataset = []
        monolingual_file = os.path.join(self.dir_name, self.monolingual_file)
        with open(monolingual_file) as f:
            for i, line in enumerate(tqdm(f.readlines())):
                object = json.loads(line)
                if args.mono_min_prob <= object['prob'] <= 1 and len(object['snippet']) > 4:
                    if self.name == 'django':
                        canonical_code = self.__class__.canonicalize_code(object['snippet'])
                    else:
                        canonical_code = object['snippet']
                    dataset.append({'intent': object['intent'], 'snippet': canonical_code})
        return dataset



