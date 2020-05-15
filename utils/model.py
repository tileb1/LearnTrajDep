#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x

        return y


class TimeAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, activation=nn.SELU()):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            activation,
            nn.Linear(32, 30),
            activation,
            nn.Linear(30, hidden_size))

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 30),
            activation,
            nn.Linear(30, 32),
            activation,
            nn.Linear(32, input_size))
    # def __init__(self, input_size, hidden_size, activation=nn.SELU()):
    #     super().__init__()
    #     self.encoder = nn.Sequential(
    #         nn.Linear(input_size, input_size),
    #         activation,
    #         nn.Linear(input_size, input_size))
    #
    #     self.decoder = nn.Sequential(
    #         nn.Linear(input_size, input_size),
    #         activation,
    #         nn.Linear(input_size, input_size))

    def forward(self, x):
        embedding = self.encoder(x)
        out = self.decoder(embedding)
        return out, embedding


class MultipleGCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, autoencoder1, autoencoder2,
                 opt, num_stage=1, node_n=48):
        super().__init__()
        self.opt = opt

        # No downsampling
        self.GCN1 = GCN(input_feature, hidden_feature, p_dropout, num_stage=num_stage, node_n=node_n)
        self.autoencoder1 = autoencoder1
        for param in self.autoencoder1.parameters():
            param.requires_grad = False

        # Downsampling by factor 2
        self.GCN2 = GCN(input_feature, hidden_feature, p_dropout, num_stage=num_stage, node_n=node_n)
        self.autoencoder2 = autoencoder2
        for param in self.autoencoder2.parameters():
            param.requires_grad = False

        # Final layer
        self.final_layer = nn.Linear(self.opt.input_n + self.opt.output_n + self.opt.output_n//2,
                                     self.opt.intput_n + self.opt.output_n)

    def forward(self, x1, x2):
        y1 = self.autoencoder1.decoder(self.GCN1(self.autoencoder1.encoder(x1)))
        y2 = self.autoencoder2.decoder(self.GCN2(self.autoencoder2.encoder(x2)))

        # Combine both outputs
        merged = torch.cat((y1, y2[:, :, self.opt.input_n:self.opt.input_n+self.opt.output_n//2]), dim=2)

        return y1, y2, self.final_layer(merged)




class IdentityAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = lambda x: x

        self.decoder = lambda x: x

    def forward(self, x):
        return x, x
