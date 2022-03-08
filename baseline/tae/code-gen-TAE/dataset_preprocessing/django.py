# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright 2018 Pengcheng Yin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#####################################################################################
# Code is based on https://github.com/pcyin/tranX/blob/master/datasets/django/dataset.py
####################################################################################


from __future__ import print_function
import json
import re
import ast
import astor
import nltk
import os
from utils import my_annotate
from .monolingual_python import MonolingualPython


p_elif = re.compile(r'^elif\s?')
p_else = re.compile(r'^else\s?')
p_try = re.compile(r'^try\s?')
p_except = re.compile(r'^except\s?')
p_finally = re.compile(r'^finally\s?')
p_decorator = re.compile(r'^@.*')
QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")


class Django(MonolingualPython):
    def __init__(self, name, split, tokenizer, args, monolingual=False):
        self.threshold = {'train': 100,
                          'dev': 100,
                          'test': 200}
        super(Django, self).__init__('django', split, tokenizer, args, monolingual)

    @staticmethod
    def canonicalize_code(code):
        if p_elif.match(code):
            code = 'if True: pass\n' + code

        if p_else.match(code):
            code = 'if True: pass\n' + code

        if p_try.match(code):
            code = code + 'pass\nexcept: pass'
        elif p_except.match(code):
            code = 'try: pass\n' + code
        elif p_finally.match(code):
            code = 'try: pass\n' + code

        if p_decorator.match(code):
            code = code + '\ndef dummy(): pass'

        if code[-1] == ':':
            code = code + 'pass'

        return code

    @staticmethod
    def canonicalize_str_nodes(py_ast, slot_map):
        str_map = {x: slot_name for slot_name, x in list(slot_map.items())}
        for node in ast.walk(py_ast):
            if isinstance(node, ast.Str):
                str_val = node.s

                if str_val in str_map:
                    node.s = str_map[str_val]
                else:
                    # handle cases like `\n\t` in string literals
                    for str_literal, slot_id in str_map.items():
                        str_literal_decoded = str_literal.encode().decode('unicode_escape')
                        if str_literal_decoded == str_val:
                            node.s = slot_id

    @staticmethod
    def canonicalize_query(query):
        """
        canonicalize the query, replace strings to a special place holder
        """
        str_count = 0
        str_map = dict()

        matches = QUOTED_STRING_RE.findall(query)
        # de-duplicate
        cur_replaced_strs = set()

        slot_map = dict()
        for match in matches:
            # If one or more groups are present in the pattern,
            # it returns a list of groups
            quote = match[0]
            str_literal = match[1]
            quoted_str_literal = quote + str_literal + quote

            if str_literal in cur_replaced_strs:
                # replace the string with new quote with slot id
                query = query.replace(quoted_str_literal, str_map[str_literal])
                continue

            # FIXME: substitute the ' % s ' with
            if str_literal in ['%s']:
                continue

            str_repr = 'str%d' % str_count
            str_map[str_literal] = str_repr

            slot_map[str_repr] = str_literal

            query = query.replace(quoted_str_literal, str_repr)

            str_count += 1
            cur_replaced_strs.add(str_literal)

        # tokenize
        query_tokens = nltk.word_tokenize(query)

        new_query_tokens = []
        # break up function calls like foo.bar.func
        for token in query_tokens:
            new_query_tokens.append(token)
            i = token.find('.')
            if 0 < i < len(token) - 1:
                new_tokens = ['['] + token.replace('.', ' . ').split(' ') + [']']
                new_query_tokens.extend(new_tokens)

        query = ' '.join(new_query_tokens)
        query = query.replace('\' % s \'', '%s').replace('\" %s \"', '%s')

        return query, slot_map

    @staticmethod
    def canonicalize_example(query, code):

        canonical_query, str_map = Django.canonicalize_query(query)
        query_tokens = canonical_query.split(' ')

        canonical_code = Django.canonicalize_code(code)
        ast_tree = ast.parse(canonical_code)

        Django.canonicalize_str_nodes(ast_tree, str_map)
        canonical_code = astor.to_source(ast_tree)
        return ' '.join(query_tokens), canonical_code, str_map

    def _download_dataset(self):
        self._download_file('https://raw.githubusercontent.com/odashi/ase15-django-dataset/master/django/all.anno', 'all.anno')
        self._download_file('https://raw.githubusercontent.com/odashi/ase15-django-dataset/master/django/all.code', 'all.code')

    def _preprocess(self):
        data = []
        json_file = os.path.join(self.dir_name, '{}.json'.format(self.split))
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
            return data
        self.anno_file_name = os.path.join(self.dir_name, 'all.anno')
        self.code_file_name = os.path.join(self.dir_name, 'all.code')
        with open(self.anno_file_name) as anno_file, open(self.code_file_name) as code_file:
            codes = code_file.readlines()
            annos = anno_file.readlines()
            for i in range(len(annos)):
                annot = annos[i].lower().strip()
                code = codes[i].lower().strip()
                annot, code, str_map = self.canonicalize_example(annot, code)
                instance = {'intent': my_annotate(annot), 'snippet': code.lower(), 'slot_map': str_map}
                if self.split == 'train' and 0 <= i < 16000:
                    data.append(instance)
                elif self.split == 'dev' and 16000 <= i < 17000:
                    data.append(instance)
                elif self.split == 'test' and 17000 <= i:
                    data.append(instance)
        with open(json_file, 'w') as f:
            json.dump(data, f)
        return data

