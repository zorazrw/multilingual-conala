# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2018,	Catherine Finegan-Dollak and Jonathan K. Kummerfeld
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
#####################################################################################
# Code is based on https://github.com/jkkummerfeld/text2sql-data/blob/master/tools/json_to_flat.py
####################################################################################


import os
import json
import re
from .monolingual_sql import MonolingualSQL


class SmallSQL(MonolingualSQL):

    def __init__(self, name, split, tokenizer, args, monolingual=False):
        self.threshold = {'train': 300,
                          'dev': 300,
                          'test': 300}
        super(SmallSQL, self).__init__(name, split, tokenizer, args, monolingual)

    def _download_dataset(self):
        if self.name == 'geography':
            self._download_file(url='https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/master/data/geography.json',
                                file_name='geography.json')
        elif self.name == 'atis':
            self._download_file(url='https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/master/data/atis.json',
                                file_name='atis.json')

    def _preprocess(self):
        json_file = os.path.join(self.dir_name, '{}.json'.format(self.split))
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                return json.load(f)
        whole_data = []
        data = json.loads(open(os.path.join(self.dir_name, '{}.json'.format(self.name))).read())
        dataset_size = 0
        for instance in data:
            if instance['query-split'] == self.split:
                dataset_size += len(instance['sentences'])
                var_sql = instance["sql"][0]
                for sentence in instance["sentences"]:
                    text = sentence['text']
                    sql = var_sql  # Needed to do variable replacement correctly
                    # Variable replacement
                    for name in sentence['variables']:
                        value = sentence['variables'][name]
                        if len(value) == 0:
                            for variable in instance['variables']:
                                if variable['name'] == name:
                                    value = variable['example']
                        text = value.join(text.split(name))
                        sql = value.join(sql.split(name))
                    sql = SmallSQL.chunk_sql(sql)
                    whole_data.append({'intent': text.lower().strip(), 'snippet': sql.lower().strip()})
        print(self.split, dataset_size)
        assert len(whole_data) == dataset_size
        with open(json_file, 'w') as f:
            json.dump(whole_data, f)
        return whole_data

    @staticmethod
    def chunk_sql(code):
        code = re.sub(r'(\w*%s\w*)' % 'alias', '', code).replace('.', '').replace('AS', '').replace(';', '')
        code = re.sub(r'\s+', ' ', code)
        code = code.replace('"', '')
        code = code.replace('\'', '')
        return code

