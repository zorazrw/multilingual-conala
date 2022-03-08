# BSD 3-Clause License
#
# Copyright (c) 2017, Salesforce Research
# All rights reserved.
# LICENSE file in dataset_preporcessing/wikisql/LICENSE
#####################################################################################
# Code is based on https://github.com/salesforce/WikiSQL/blob/master/annotate.py
####################################################################################

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import ujson as json
from stanza.server.client import CoreNLPClient
from tqdm import tqdm
from dataset_preprocessing.wikisql.lib.common import count_lines, detokenize
from utils import my_annotate

client = None

def annotate(sentence, lower=True):
    global client
    if client is None:
        client = CoreNLPClient(default_annotators='ssplit,tokenize'.split(','))
    words, gloss, after = [], [], []
    for s in client.annotate(sentence).sentence:
        for t in s.token:
            words.append(t.word)
            gloss.append(t.originalText)
            after.append(t.after)
    return {
        'gloss': gloss,
        'words': words,
        'after': after,
        }


def annotate_example(example, table):
    ann = {'table_id': example['table_id']}
    ann['question'] = my_annotate(example['question'])
    ann['table'] = {
        'header': [my_annotate(h) for h in table['header']],
        'type': table['types'],
    }
    ann['query'] = example['sql']
    for c in ann['query']['conds']:
        c[-1] = my_annotate(str(c[-1]))
    return ann


def is_valid_example(e):
    if not all([h['words'] for h in e['table']['header']]):
        return False
    headers = [detokenize(h).lower() for h in e['table']['header']]
    if len(headers) != len(set(headers)):
        return False
    input_vocab = set(e['seq_input']['words'])
    for w in e['seq_output']['words']:
        if w not in input_vocab:
            print('query word "{}" is not in input vocabulary.\n{}'.format(w, e['seq_input']['words']))
            return False
    input_vocab = set(e['question']['words'])
    for col, op, cond in e['query']['conds']:
        for w in cond['words']:
            if w not in input_vocab:
                print('cond word "{}" is not in input vocabulary.\n{}'.format(w, e['question']['words']))
                return False
    return True


def create_annotations(din, dout):
    if not os.path.isdir(dout):
        os.makedirs(dout)

    for split in ['train', 'dev', 'test']:
        fsplit = os.path.join(din, split) + '.jsonl'
        ftable = os.path.join(din, split) + '.tables.jsonl'
        fout = os.path.join(dout, split) + '_annotated.jsonl'

        print('annotating {}'.format(fsplit))
        with open(fsplit) as fs, open(ftable) as ft, open(fout, 'wt') as fo:
            print('loading tables')
            tables = {}
            for line in tqdm(ft, total=count_lines(ftable)):
                d = json.loads(line)
                tables[d['id']] = d
            print('loading examples')
            n_written = 0
            for line in tqdm(fs, total=count_lines(fsplit)):
                d = json.loads(line)
                a = annotate_example(d, tables[d['table_id']])
                fo.write(json.dumps(a) + '\n')
                n_written += 1
            print('wrote {} examples'.format(n_written))


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--din', default='data/wikisql', help='data directory')
    parser.add_argument('--dout', default='data/wikisql', help='output directory')
    args = parser.parse_args()
    create_annotations(args.din, args.dout)

