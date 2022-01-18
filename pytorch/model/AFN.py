# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:09:45 2022

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn


class LogTransformLayer(nn.Module):
    """
        Logarithmic Transformation Layer in AFN
    """
    
    def __init__(self, field_size, embed_size, hidden_size):
        super(LogTransformLayer, self).__init__()
        
        self.weights = nn.Parameter(torch.Tensor(field_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(1,1,hidden_size))
        nn.init.normal_(self.weights, mean=0, std=0.1) # 初始化,否则神经元容易dead
        
        self.bn1 = nn.BatchNorm1d(embed_size)
        self.bn2 = nn.BatchNorm1d(embed_size)
        
    def forward(self, x):
        """
            x : batch_size, field_size, embed_size
        """
        # log输入要求不能有负数, 所以embedding层的内容都改为正数, 0改为很小的正数
        tmp = torch.abs(x)
        tmp = torch.clamp(tmp, min=1e-6)
        # Transpose
        tmp = torch.transpose(tmp, 1, 2) # batch_size, embed_size, field_size
        # logarithmic transfom
        tmp = torch.log(tmp) # 取对数
        tmp = self.bn1(tmp)
        tmp = torch.matmul(tmp, self.weights) + self.bias # batch, embed, hidden
        tmp = torch.exp(tmp) # 取幂
        tmp = self.bn2(tmp)
        tmp = tmp.flatten(start_dim=1) # batch, embed * hidden
        
        return tmp


class AFN(nn.Module):
    """
        Adaptive Factorization Network
    """
    def __init__(self, feature_fields, embed_size, hidden_size, mlp_dims = [128, 64], dropout = 0.1):
        super(AFN, self).__init__()
        self.field_size = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)

        #First Order
        self.linear = torch.nn.Embedding(sum(feature_fields)+1, 1)
        self.bias = torch.nn.Parameter(torch.zeros((1,)))     
        
        # Embedding
        self.embedding = torch.nn.Embedding(sum(feature_fields)+1, embed_size)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        
        # AFM Layer
        self.LogTransformLayer = LogTransformLayer(field_size = self.field_size, 
                                                   embed_size = embed_size, 
                                                   hidden_size = hidden_size)
        # DNN Layer
        layers = []
        input_dim = embed_size * hidden_size
        for mlp_dim in mlp_dims:
            # 全连接层
            layers.append(nn.Linear(input_dim, mlp_dim))
            layers.append(nn.BatchNorm1d(mlp_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p = dropout))
            input_dim = mlp_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
 
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0)
        embed_x = self.embedding(tmp)
        # First Order
        linear_part = torch.sum(self.linear(tmp), dim = 1) + self.bias        
        
        # Logtransform + DNN
        afn_part = self.LogTransformLayer(embed_x)
        afn_part = self.mlp(afn_part)
        res = linear_part + afn_part
        res = torch.sigmoid(res.squeeze(1))
        
        return res
        
        
        
        
        
        
        