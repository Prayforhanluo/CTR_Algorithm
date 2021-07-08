# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 18:29:57 2021

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn

class FactorizationMachine(nn.Module):
    """
        Factorization Machine
    """
    def __init__(self, feature_fields, embed_dim):
        """
            feature_fileds : array_like
                             类别特征的field的数目
        """
        super(FactorizationMachine, self).__init__()
        
        #输入的是label coder 用输出为1的embedding来形成linear part
        self.linear = torch.nn.Embedding(sum(feature_fields)+1, 1)
        self.bias = torch.nn.Parameter(torch.zeros((1,)))
        
        self.embedding = torch.nn.Embedding(sum(feature_fields)+1, embed_dim)
        self.offset = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        nn.init.xavier_uniform_(self.embedding.weight.data)
        
    def forward(self, x):
        tmp = x + x.new_tensor(self.offset).unsqueeze(0)
        
        # 线性层
        linear_part = torch.sum(self.linear(tmp), dim = 1) + self.bias
        
        #内积项
        ## embedding
        tmp = self.embedding(tmp)
        ##  XY
        square_of_sum = torch.sum(tmp, dim=1) ** 2
        sum_of_square = torch.sum(tmp ** 2, dim=1)
        
        x = linear_part + 0.5 * torch.sum(square_of_sum - sum_of_square, dim = 1, keepdim=True)
        # sigmoid
        x = torch.sigmoid(x.squeeze(1))
        return x