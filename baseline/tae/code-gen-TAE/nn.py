# Copyright (c) 2020-present, Royal Bank of Canada.
# From PyTorch: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoder

import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerDecoderLayer, TransformerDecoder
import torch


class MyTransformerDecoder(TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTransformerDecoder, self).__init__(decoder_layer, num_layers, norm)

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, no_context_update=False, no_memory=False):
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    no_context_update=no_context_update, no_memory=no_memory)

        if self.norm:
            output = self.norm(output)

        return output


class MyTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(MyTransformerDecoderLayer, self).__init__(d_model=d_model, nhead=nhead,
                                                              dim_feedforward=dim_feedforward, dropout=dropout,
                                                              activation=activation)

    def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                no_context_update=False, no_memory=False):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if not no_memory:
            if no_context_update:
                with torch.no_grad():
                    tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                               key_padding_mask=memory_key_padding_mask)[0]
            else:
                tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                           key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def generate_square_subsequent_mask(sz: int):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
