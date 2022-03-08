# BSD 3-Clause License
#
# Copyright (c) 2017, Salesforce Research
# All rights reserved.
# LICENSE file in dataset_preporcessing/wikisql/LICENSE
#####################################################################################
# Code is based on https://github.com/salesforce/WikiSQL/blob/master/lib/common.py
####################################################################################

def count_lines(fname):
    with open(fname) as f:
        return sum(1 for line in f)


def detokenize(tokens):
    ret = ''
    for g, a in zip(tokens['gloss'], tokens['after']):
        ret += g + a
    return ret.strip()
