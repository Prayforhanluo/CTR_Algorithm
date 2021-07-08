# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:41:39 2021

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn

class AttentionalFactorizationMachine(nn.Module):
    """
        Attentional FM
        在FM的基础上给各个交叉特征赋予了attention的权重，增加了可解释性
    """
    def __init__(self, feature_fields, embed_dim, attn_size, dropouts):
        super(AttentionalFactorizationMachine, self).__init__()
        self.num_fields = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        
        #线性部分
        self.linear = torch.nn.Embedding(sum(feature_fields)+1, 1)
        self.bias = torch.nn.Parameter(torch.zeros((1,)))  
        
        #embedding
        self.embedding = torch.nn.Embedding(sum(feature_fields)+1, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        
        #attention部分
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts
    
    def forward(self, x):
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0) #构造成embeding形式
        linear_part = torch.sum(self.linear(tmp), dim = 1) + self.bias # 线性部分
        
        tmp = self.embedding(tmp) # embedding后的vec
        
        # 交叉项, 并加入attention
        num_fields = tmp.shape[1]
        row, col = [], []
        for i in range(num_fields - 1):
            for j in range(i+1, num_fields):
                row.append(i)
                col.append(j)
        p, q = tmp[:, row], tmp[:,col]
        inner = p * q
        attn_scores = nn.functional.relu(self.attention(inner))
        attn_scores = nn.functional.softmax(self.projection(attn_scores), dim=1)
        attn_scores = nn.functional.dropout(attn_scores, p = self.dropouts[0])
        attn_output = torch.sum(attn_scores * inner, dim = 1)
        attn_output = nn.functional.dropout(attn_output, p = self.dropouts[1])
        inner_attn_part = self.fc(attn_output)
        
        # 最后输出
        x = linear_part + inner_attn_part
        x = torch.sigmoid(x.squeeze(1))
        return x