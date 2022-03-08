# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from evaluation.search import beam_search, greedy_search
from tqdm import tqdm
from utils import compute_loss


def evaluate(args, valid_loader, model, split='dev'):
    model.eval()
    with torch.no_grad():
        averaged_loss = 0
        for i, data in enumerate(tqdm(valid_loader)):
            loss, _, _ = compute_loss(args, data, model)
            averaged_loss += loss.sum()

        averaged_loss = (averaged_loss / len(valid_loader.dataset)).item()
        print('average {} loss:'.format(split), averaged_loss)
        return averaged_loss


def generate_hypothesis(args, valid_loader, model, search):
    whole_hype = []
    for i, data in enumerate(tqdm(valid_loader)):
        data['target']['input_ids'] *= 0
        if args.pointer_network:
            data['label'] *= 0

        prediction_length = data['target']['input_ids'].shape[1] + 25
        if search == 'beam':
            predicted_query = beam_search(args, model, data,
                                          prediction_length=prediction_length)
        elif search == 'greedy':
            predicted_query = greedy_search(args, model, data,
                                            prediction_length=prediction_length)
        sep_text = '</s>' if args.use_codebert else '[SEP]'
        for pred in predicted_query:
            if search == 'beam':
                current_hypes = []
                for p in pred:
                    hypothesis = model.tokenizer.decode(p)
                    end_index = hypothesis.find(sep_text)
                    #if end_index==-1:
                    #    print("not found end of setnence")
                    hypothesis = hypothesis[:end_index]
                    current_hypes.append({'str': hypothesis, 'token': p})
                whole_hype.append(current_hypes)
            else:
                hypothesis = model.tokenizer.decode(pred)
                end_index = hypothesis.find(sep_text)
                #if end_index == -1:
                #    print("not found end of setnence")
                hypothesis = hypothesis[:end_index]
                whole_hype.append({'str': hypothesis, 'token': pred})

    return whole_hype
