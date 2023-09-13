# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
from config import cfg
from torch_geometric.nn import GATConv, BatchNorm


def make_linear_layers(feat_dims, relu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and relu_final):
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
            ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.ConvTranspose2d(  # (input-1)*stride+ output_padding -2*padding+kernel_size, (input-1)*2+2,->16,32,64
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_GAT_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            GATConv(feat_dims[i], feat_dims[i + 1]))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(BatchNorm(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


def make_upsample_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            Interpolate(2, 'bilinear'))
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=3,
                stride=1,
                padding=1
            ))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(ResBlock, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.conv = make_conv_layers([in_feat, out_feat, out_feat], bnrelu_final=False)
        self.bn = nn.BatchNorm2d(out_feat)
        if self.in_feat != self.out_feat:
            self.shortcut_conv = nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=1, padding=0)
            self.shortcut_bn = nn.BatchNorm2d(out_feat)

    def forward(self, input):
        x = self.bn(self.conv(input))
        if self.in_feat != self.out_feat:
            x = F.relu(x + self.shortcut_bn(self.shortcut_conv(input)))
        else:
            x = F.relu(x + input)
        return x


def make_conv3d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv3d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
            ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm3d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_deconv3d_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.ConvTranspose3d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm3d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)  # shape [N, out_features]
        node_num = h.size()[1]
        # 生成N*N的嵌入
        attention_input = torch.cat([h.repeat_interleave(node_num, dim=1), h.repeat(1, node_num, 1)], dim=2).view(-1, node_num, node_num, 2*self.out_features)
        e = self.leakyrelu(torch.matmul(attention_input, self.a).squeeze(3))

        zero_vec = -9e15*torch.ones_like(e)
        adj = adj.repeat(e.size()[0], 1, 1)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class BGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=1):
        super(BGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        residual = x

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

