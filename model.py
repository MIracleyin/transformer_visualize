#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/6/1 14:38
# !@Author  : miracleyin @email: miracleyin@live.com
# !@file: model.py

import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import display_tensorshape

display_tensorshape()

class RNNModel(nn.Module):
    """
    Language model is composed of three parts:
    a word embedding layer, a rnn network and a output layer
    The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    """
    def __init__(self, nvoc, ninput, nhid, nlayers, drop=0.5, init_uniform=0.1):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(drop)
        self.encoder = nn.Embedding(nvoc, ninput)
        self.rnn = nn.LSTM(input_size=ninput, hidden_size=nhid, num_layers=nlayers)
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights(init_uniform)

    def init_weights(self, init_uniform):
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        embeddings = self.encoder(input)
        embeddings = self.drop(embeddings)
        output, hidden = self.rnn(embeddings)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, initrange=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights(initrange)

    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self, initrange):
        encoder_weight = self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        decoder_weight = self.decoder.weight.data.uniform_(-initrange, initrange)
        return encoder_weight, decoder_weight

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output_attention = self.transformer_encoder(src, src_mask)
        output = self.decoder(output_attention)

        return F.log_softmax(output, dim=-1), output_attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (- math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0),:]
        return self.dropout(x)

if __name__ == '__main__':
    rnn = RNNModel(1000,64,64,3)
    a = torch.zeros([1000,1]).long()
    b1, b2 = rnn(a)
    print(a.shape)