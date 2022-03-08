# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BertModel
from torch.nn import LayerNorm
from nn import MyTransformerDecoder, MyTransformerDecoderLayer, generate_square_subsequent_mask
import torch


class Model(nn.Module):
    def __init__(self, pretrained_weights, args):
        super(Model, self).__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(pretrained_weights)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)
        self.myembedding = MyEmbedding(self.encoder.embeddings)
        config = self.encoder.config
        if args.random_encoder:
            self.encoder = BertModel(config)
        elif args.no_encoder:
            self.encoder = None
        print(config)
        decoder_layer = MyTransformerDecoderLayer(config.hidden_size, config.num_attention_heads,
                                                config.intermediate_size, dropout=0.1, activation='gelu')
        self.decoder = MyTransformerDecoder(decoder_layer, num_layers=args.decoder_layers, norm=LayerNorm(config.hidden_size))
        self.device = args.device
        self.copy_attention = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                            nn.Tanh() if not args.use_gelu else nn.GELU(),
                                            nn.Linear(config.hidden_size, config.hidden_size))
        self.linear_before_softmax = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                   nn.Tanh() if not args.use_gelu else nn.GELU(),
                                                   nn.Linear(config.hidden_size, config.hidden_size))

    def encode(self, source, no_context_update):
        encoder_output = self.encoder(
                     input_ids=source['input_ids'] if (not (self.args.dummy_source and no_context_update)) else source['input_ids'] * 0,
                     attention_mask=source['attention_mask'],
                     token_type_ids=source['token_type_ids'] if self.args.translate_backward is False else None)[0]
        if self.args.extra_encoder:
            encoder_output = self.extra_encoder(src=torch.transpose(encoder_output, 0, 1),
                                                mask=None,
                                                src_key_padding_mask=(source['attention_mask'] == 0))
            encoder_output = torch.transpose(encoder_output, 0, 1)
        return encoder_output

    def forward(self, data, target_input=None, no_encoder=None, no_context_update=False, return_encoder_output=False, encoder_output_saved=None):
        source = {key: value.to(self.device) for key, value in data['source'].items()}
        target = {key: value.to(self.device) for key, value in data['target'].items()}
        label, choices = None, None
        if self.args.pointer_network:
            choices = {key: value.to(self.device) for key, value in data['choices'].items()}
            label = data['label'].to(self.device)
        if target_input is not None:
            target = target_input
        if encoder_output_saved is not None:
            encoder_output = encoder_output_saved
        elif not self.args.no_encoder:
            if no_context_update:
                with torch.no_grad():
                    encoder_output = self.encode(source, no_context_update)
            else:
                encoder_output = self.encode(source, no_context_update) # if not self.args.translate_backward else None)[0]
            # encoder_output *= 0

        if return_encoder_output:
            return encoder_output

        target_embedding = self.myembedding(target['input_ids'][:, :-1])
        target_length = target['input_ids'].shape[1]
        prediction = self.decoder(tgt=torch.transpose(target_embedding, 0, 1),
                                  memory=torch.transpose(encoder_output, 0, 1) if not self.args.no_encoder else None,
                                  tgt_mask=generate_square_subsequent_mask(target_length - 1).to(self.device),
                                  memory_mask=None,
                                  tgt_key_padding_mask=target['attention_mask'][:, :-1] == 0,
                                  memory_key_padding_mask=(source['attention_mask'] == 0) if not self.args.no_encoder else None,
                                  no_memory=self.args.no_encoder,
                                  no_context_update=False
                                  )
        prediction = torch.transpose(prediction, 0, 1)
        generation_prediction = self.linear_before_softmax(prediction)

        if self.args.pointer_network:
            choices_emb = self.myembedding.pembedding.word_embeddings(choices['input_ids'])
            logits = torch.einsum('bid, bjd->bij', prediction, choices_emb)
            logits = logits.masked_fill_(
                (choices['attention_mask'] == 0).unsqueeze(1).expand(-1, logits.shape[1], -1), float('-inf'))
        else:
            logits = torch.matmul(generation_prediction, torch.t(self.myembedding.pembedding.word_embeddings.weight))

        if not self.args.no_encoder and self.args.use_copy_attention:
            copy_prediction = self.copy_attention(prediction)
            copy_attention = torch.einsum('bid, bjd->bij', copy_prediction, encoder_output)
            if self.args.pointer_network:
                index = source['source_label']
            else:
                index = source['input_ids']
            copy_attention = copy_attention.masked_fill_(
                (source['attention_mask'] == 0).unsqueeze(1).expand(-1, copy_attention.shape[1], -1), 0)
            logits.scatter_add_(index=index.unsqueeze(1).expand(-1, logits.shape[1], -1),
                                src=copy_attention, dim=2)

        return logits, target, choices, label, prediction


class MyEmbedding(nn.Module):
    def __init__(self, embedding):
        super(MyEmbedding, self).__init__()
        self.pembedding = embedding

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.pembedding.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.pembedding.word_embeddings(input_ids)
        position_embeddings = self.pembedding.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.pembedding.LayerNorm(embeddings)
        embeddings = self.pembedding.dropout(embeddings)
        return embeddings
