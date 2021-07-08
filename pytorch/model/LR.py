# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 18:23:10 2021

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """
        LR
    """
    def __init__(self, feature_fields):
        super(LogisticRegression, self).__init__()
        self.feature_fields = feature_fields
        
        #输入的是labelencoder的矩阵 用输出为1的embedding来形成linear part
        self.linear = torch.nn.Embedding(sum(feature_fields)+1, 1)
        self.bias = torch.nn.Parameter(torch.zeros((1,)))     
        self.offset = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        
    def forward(self, x):
        x = x + x.new_tensor(self.offset).unsqueeze(0)
        x = torch.sum(self.linear(x), dim = 1) + self.bias
        x = torch.sigmoid(x.squeeze(1))
        return x

