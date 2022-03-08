# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#####################################################################################
# Code is based on https://github.com/pcyin/tranX/blob/dca34e813f2f6ebcb5dd29605e8fb1ec75c4a9bd/datasets/conala/bleu_score.py
####################################################################################

import re
from dataset_preprocessing.conala import Conala
from nltk.translate.bleu_score import SmoothingFunction
import nltk
import collections
import math
from utils import my_detokenize_code
import astor
import ast


def _get_ngrams(segment, max_order):
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def tokenize_for_bleu_eval_python(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens


def tokenize_for_bleu_java(code):
    code = re.sub(r'([^A-Za-z0-9])', r' \1 ', code)  # split by punct
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)  # split camel
    code = re.sub(r'([0-9])([A-Z])', r'\1 \2', code)  # split camel
    code = re.sub(r'(# )+', '# ', code)  # new lines to new line
    code = re.sub(r'\. # ', '.', code)
    code = re.sub(r'\s+', ' ', code)  # spaces to single space
    code = code.replace('"', '`')  # quote
    code = code.replace('\'', '`')  # quote
    return code


def tokenize_for_bleu_eval_sql(code):
    code = re.sub(r'([^A-Za-z0-9])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'([0-9])([A-Z])', r'\1 \2', code)
    code = code.replace('"', '')
    code = code.replace('\'', '')
    code = code.replace(";", '')
    code = re.sub(r'\s+', ' ', code)
    code = code.lower()
    tokens = [t for t in code.split(' ') if t]
    return tokens


def remove_s(string):
    string = re.sub(r'( _ _ )([a-z]+)( _ _ )', r'__\2__', string)
    string = re.sub(r'([a-z]+)( _ )([a-z]+)', r'\1_\3', string)
    return string


def remove_spaces(mistake, intent):
    detokenized_code = my_detokenize_code(mistake, intent)
    final_code = detokenized_code.replace("* *", "**").replace('+ =', '+=').replace('- =', '-=').replace('! =', '!=') \
            .replace('= =', '==').replace('< =', '<=').replace('> =', '>=').replace('/ /', '//').replace('except', '\nexcept') \
            .replace('finally', '\nfinally').replace('< <', '<<').replace('> >', '>>').replace('elif', '\nelif')
    no_space = final_code.replace(' _ ', '_').replace('_ ', '_').replace(' _', '_')
    try:
        tree = ast.parse(no_space)
        return astor.to_source(tree).strip()
    except SyntaxError:
        final_code_valid =False
        try:
            ast.parse(final_code)
            final_code_valid=True
            ast.parse(no_space)
        except:
            if final_code_valid == True:
                pass
                #print("syntax error")
                #print(final_code)
                #print(no_space)
        return final_code


def compute_bleu(translation_corpus, dataset_object, section, args, max_order=4, smooth=False):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    bleu_sentence = 0

    for index in section:
        translation = translation_corpus[index]
        if args.translate_backward:
            reference = ' '.join(dataset_object[index]['intent']['words'])
        else:
            reference = dataset_object[index]['snippet']

        if args.python is True and args.translate_backward is False:
            slot_map = dataset_object[index]['slot_map']
            ref = Conala.decanonicalize_code(reference, slot_map)
            hype = remove_spaces(Conala.decanonicalize_code(translation, slot_map), dataset_object[index]['intent'])
            # print('ref', ref)
            # print('hype', hype)
            reference = tokenize_for_bleu_eval_python(ref)
            translation = tokenize_for_bleu_eval_python(hype)
        elif args.dataset_name=='magic':
            reference = tokenize_for_bleu_java(reference)
            translation = tokenize_for_bleu_java(translation)
        else:
            reference = tokenize_for_bleu_eval_sql(reference)
            translation = tokenize_for_bleu_eval_sql(translation)
        references = [reference]
        bleu_sentence += nltk.translate.bleu_score.sentence_bleu(references,
                                                                 translation,
                                                                 smoothing_function=SmoothingFunction().method3)

        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        if ratio == 0.:
            bp = 0.
        else:
            bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp
    return bleu, bleu_sentence/len(section)

