import math
import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

import config


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, A, D):
        output = D.mm(A).mm(D).mm(input).mm(self.weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphLearning(Module):
    def __init__(self, in_features):
        super(GraphLearning, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.FloatTensor(1, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        s = torch.zeros((len(inputs), len(inputs)))
        for i in range(len(inputs)):
            for j in range(len(inputs)):
                # print(torch.exp(F.relu(self.weight.mm(torch.abs(inputs[i] - inputs[j]).unsqueeze(0).t())))[0][0])
                # print(s[i, j])
                # exit()
                s[i, j] = torch.exp(F.relu(self.weight.mm(torch.abs(inputs[i] - inputs[j]).unsqueeze(0).t())))[0][0]
        A = F.softmax(s, dim=1)
        D = torch.diag(torch.sum(A, dim=1))
        return A, D

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.in_features) + ')'


class GCN(nn.Module):
    def __init__(self, dropout):
        super(GCN, self).__init__()
        # self.graph_learning = GraphLearning(config.features_dim_num)
        self.gc1 = GraphConvolution(config.features_dim_num, config.features_dim_num)
        self.gc2 = GraphConvolution(config.features_dim_num, config.GCN_hidderlayer_dim_num)
        self.gc3 = GraphConvolution(config.GCN_hidderlayer_dim_num, config.class_num)
        self.dropout = dropout

    def forward(self, x, A, D):
        x = self.gc1(x, A, D)
        features = x
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, A, D)
        x = F.relu(x)
        x = self.gc3(x, A, D)
        return x, features
















