# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from utils import compute_loss


def greedy_search(args, model, data, prediction_length):
    cls = torch.zeros(len(data['source']['input_ids']), prediction_length).to(args.device)
    cls[:, 0] = 101
    attention_mask = cls.long()
    attention_mask[:, 0] = 1
    generated_seq = {'attention_mask': attention_mask, 'input_ids': cls.long()}
    with torch.cuda.amp.autocast():
        encoder_outputs = model(data, target_input=None, no_context_update=False, return_encoder_output=True)
    for i in range(1, prediction_length):
        target_input = ({k: v[:, :i + 1] for k, v in generated_seq.items()})
        with torch.cuda.amp.autocast():
            _, logits, choices = compute_loss(args, data, model, target_input=target_input, encoder_output_saved=encoder_outputs)
        logits = logits[:, i-1]
        selected_indices = torch.argmax(logits, dim=-1, keepdim=True)
        if args.pointer_network:
            selected_tokens = torch.gather(choices['input_ids'], index=selected_indices, dim=1).squeeze()
        else:
            selected_tokens = selected_indices.squeeze()
        generated_seq['input_ids'][:, i] = selected_tokens.squeeze()
        generated_seq['attention_mask'][:, i] = 1
    return generated_seq['input_ids'][:, 1:]


def repeat_list(input_list, counts):
    return [input_list[i // counts] for i in range(counts * len(input_list))]


def length_norm(num, alpha, base=5):
    if num == 0:
        return 0
    return ((base + num) ** alpha) / ((base + 1) ** alpha)


def beam_search(args, model, data, prediction_length=140):
    beam_num = args.beam_num
    num_questions = len(data['source']['input_ids'])
    for key in data['source'].keys():
        data['source'][key] = torch.stack(repeat_list(data['source'][key], beam_num), dim=0)
    cls = torch.zeros(num_questions * beam_num, prediction_length).to(args.device)
    cls[:, 0] = 101
    scores = torch.zeros(num_questions * beam_num)
    attention_mask = cls.clone().long()
    attention_mask[:, 0] = 1
    generated_seq = {'attention_mask': attention_mask, 'input_ids': cls.clone().long()}
    score_pools = [[] for _ in range(num_questions)]
    seq_pools = [[] for _ in range(num_questions)]
    if args.pointer_network:
        SEP = 2
    else:
        SEP = 102
    with torch.cuda.amp.autocast():
        encoder_outputs = model(data, target_input=None, no_context_update=False, return_encoder_output=True)
    torch.cuda.empty_cache()
    for i in range(1, prediction_length):
        target_input = ({k:v[:, :i+1]  for k, v in generated_seq.items()})
        with torch.cuda.amp.autocast():
            _, logits, choices = compute_loss(args, data, model, target_input=target_input, encoder_output_saved=encoder_outputs)
        if args.pointer_network:
            choices['input_ids'] = choices['input_ids'].to('cpu')
        logits = logits[:, i - 1]
        sep_score = logits[:, SEP].clone()
        pf = torch.logsumexp(logits, keepdim=True, dim=-1)
        logits[:, SEP] = -float('inf')
        logits, topKselected = torch.topk(logits, dim=-1, k=(beam_num))
        loss = -logits +pf
        sep_score = -sep_score + pf.squeeze()
        sep_score = sep_score.to('cpu')
        topKselected = topKselected.to('cpu')
        l_i_ = length_norm(i - 1, alpha=args.beam_search_alpha, base=args.beam_search_base)
        l_i = length_norm(i, alpha=args.beam_search_alpha, base=args.beam_search_base)
        loss = loss.to('cpu')
        temp_scores = (scores.unsqueeze(1).expand(-1, beam_num) * l_i_ + loss) / l_i
        temp_scores = temp_scores.reshape(-1, beam_num ** 2)
        if i == 1:
            indices = torch.arange(beam_num).unsqueeze(0).expand(num_questions, -1)
            scores[:] = torch.gather(temp_scores, index=indices, dim=1).squeeze().reshape(-1)
        else:
            topkscores, indices = torch.topk(temp_scores, dim=-1, k=beam_num, largest=False, sorted=True)
            finalized_scores = (scores * l_i_ + sep_score) / l_i
            finalized_index = torch.nonzero((finalized_scores < topkscores[:, -1].unsqueeze(1).expand(-1,
                                                                                                      beam_num).reshape(
                num_questions * beam_num)).float()).squeeze(1)
            if len(finalized_index) > 0:
                finalized_seq = generated_seq['input_ids'][finalized_index, 1:]
                finalized_scores = finalized_scores[finalized_index].clone()
                finalized_seq = finalized_seq.clone()
                finalized_seq[:, i - 1:] = SEP
                finalized_index = finalized_index // (beam_num)
                for q_index, q in enumerate(finalized_index):
                    score_pools[q].append(finalized_scores[q_index])
                    seq_pools[q].append(finalized_seq[q_index].to('cpu'))
            scores[:] = topkscores.reshape(-1)

        selected_indices = torch.gather(topKselected.reshape(-1, beam_num ** 2), index=indices, dim=1).reshape(-1, 1)
        if args.pointer_network:
            selected_tokens = torch.gather(choices['input_ids'], index=selected_indices, dim=1).squeeze()
        else:
            selected_tokens = selected_indices.squeeze()
        indices = indices + torch.arange(num_questions).unsqueeze(-1) * beam_num ** 2
        generated_seq['input_ids'] = generated_seq['input_ids'][(indices // beam_num).reshape(-1)]
        generated_seq['input_ids'][:, i] = selected_tokens
        generated_seq['attention_mask'][:, i] = 1
    final = []
    for q in range(num_questions):
        if len(score_pools[q]) == 0:
            final.append(generated_seq['input_ids'][q * beam_num: (q+1) * beam_num, 1:].to('cpu'))
        else:
            _, index = torch.topk(torch.tensor(score_pools[q]), largest=False, k=min(len(score_pools[q]), beam_num), sorted=True)
            final.append([seq_pools[q][ind] for ind in index])
    return final
