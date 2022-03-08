# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright 2018 Pengcheng Yin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#####################################################################################
# Code is based on https://github.com/pcyin/tranX/blob/master/datasets/conala/util.py
####################################################################################


import json
import re
import ast
import astor
import os
from .monolingual_python import MonolingualPython
from utils import my_annotate
from zipfile import ZipFile


QUOTED_TOKEN_RE = re.compile(r"(?P<quote>''|[`'\"])(?P<string>.*?)(?P=quote)")


def infer_slot_type(quote, value):
    if quote == '`' and value.isidentifier():
        return 'var'
    return 'str'


def is_enumerable_str(identifier_value):
    return len(identifier_value) > 2 and identifier_value[0] in ('{', '(', '[') and identifier_value[-1] in ('}', ']', ')')


class Conala(MonolingualPython):
    def __init__(self, name, split, tokenizer, args, monolingual=False):
        self.threshold = {'train': 100,
                          'dev': 100,
                          'test': 100}
        super(Conala, self).__init__('conala', split, tokenizer, args, monolingual)

    @staticmethod
    def decanonicalize_code(code, slot_map):

        for slot_name, slot_val in slot_map.items():
            if is_enumerable_str(slot_name):
                code = code.replace(slot_name, slot_val)

        for slot_name, slot_val in slot_map.items():
            code = code.replace(slot_name, slot_val)
        return code

    @staticmethod
    def replace_identifiers_in_ast(py_ast, identifier2slot):
        for node in ast.walk(py_ast):
            for k, v in list(vars(node).items()):
                if k in ('lineno', 'col_offset', 'ctx'):
                    continue
                if isinstance(v, str):
                    # print(str(v))
                    if v in identifier2slot:
                        slot_name = identifier2slot[v]
                        setattr(node, k, slot_name)

    @staticmethod
    def canonicalize_code(code, slot_map):
        string2slot = {x: slot_name for slot_name, x in list(slot_map.items())}
        py_ast = ast.parse(code)
        Conala.replace_identifiers_in_ast(py_ast, string2slot)
        canonical_code = astor.to_source(py_ast).strip()
        entries_that_are_lists = [slot_name for slot_name, val in slot_map.items() if is_enumerable_str(val)]
        if entries_that_are_lists:
            for slot_name in entries_that_are_lists:
                list_repr = slot_map[slot_name]
                first_token = list_repr[0]  # e.g. `[`
                last_token = list_repr[-1]  # e.g., `]`
                fake_list = first_token + ' ' + slot_name + ' ' + last_token
                slot_map[fake_list] = slot_map[slot_name]
                canonical_code = canonical_code.replace(list_repr, fake_list)
        return canonical_code

    @staticmethod
    def canonicalize_intent(intent):
        marked_token_matches = QUOTED_TOKEN_RE.findall(intent)
        slot_map = dict()
        var_id = 0
        str_id = 0
        for match in marked_token_matches:
            quote = match[0]
            value = match[1]
            quoted_value = quote + value + quote
            slot_type = infer_slot_type(quote, value)

            if slot_type == 'var':
                slot_name = 'var%d' % var_id
                var_id += 1
            else:
                slot_name = 'str%d' % str_id
                str_id += 1

            intent = intent.replace(quoted_value, slot_name)
            slot_map[slot_name] = value.strip().encode().decode('unicode_escape', 'ignore')
        return intent, slot_map

    @staticmethod
    def preprocess_example(example_json):
        intent = example_json['intent']
        if 'rewritten_intent' in example_json:
            rewritten_intent = example_json['rewritten_intent']
        else:
            rewritten_intent = None

        if rewritten_intent is None:
            rewritten_intent = intent
        rewritten_intent = rewritten_intent.lower().strip()
        snippet = example_json['snippet'].lower().strip()

        canonical_intent, slot_map = Conala.canonicalize_intent(rewritten_intent)
        canonical_snippet = Conala.canonicalize_code(snippet, slot_map)
        canonical_intent = my_annotate(canonical_intent)

        return {'intent': canonical_intent,
                'slot_map': slot_map,
                'snippet': canonical_snippet}

    def preprocess_dataset(self, file_path):
        try:
            dataset = json.load(open(file_path))
        except:
            dataset = [json.loads(jline) for jline in open(file_path).readlines()]
        examples = []
        for i, example_json in enumerate(dataset):
            try:
                example_dict = Conala.preprocess_example(example_json)
            except (AssertionError, SyntaxError, ValueError, OverflowError) as e:
                continue
            examples.append(example_dict)
        return examples

    def _download_dataset(self):
        self._download_file('http://www.phontron.com/download/conala-corpus-v1.1.zip', 'conala-corpus/conala-train.json')
        if not os.path.exists(os.path.join(self.dir_name, 'conala-corpus/conala-train.json')):
            with ZipFile(os.path.join(self.dir_name, 'conala-corpus-v1.1.zip'), 'r') as zipObj:
                zipObj.extract('conala-corpus/conala-train.json', self.dir_name)
                zipObj.extract('conala-corpus/conala-test.json', self.dir_name)

    def _preprocess(self):
        json_file = os.path.join(self.dir_name, '{}.json'.format(self.split))
        if not os.path.exists(json_file):
            examples = self.preprocess_dataset(
                os.path.join(self.dir_name, 'conala-corpus/conala-{}.json'.format(self.split if self.split != 'dev' else 'train')))
            if self.split == 'dev':
                examples = examples[-200:]
            elif self.split == 'train':
                examples = examples[:-200]
            with open(os.path.join(self.dir_name, '{}.json'.format(self.split)), 'w') as f:
                json.dump(examples, f)
            return examples
        else:
            with open(json_file) as f:
                examples = json.load(f)
            return examples
