# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 18:32:45 2021

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn

class FieldAwareFactorizationMachine(nn.Module):
    """
        FFM 
    """
    def __init__(self, field_dims, embed_dim):
        super(FieldAwareFactorizationMachine, self).__init__()
        
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        
        # 输入的是label coder 用输出为1的embedding来形成linear part
        # linear part
        self.linear = torch.nn.Embedding(sum(field_dims)+1, 1)
        self.bias = torch.nn.Parameter(torch.zeros((1,)))
        
        # ffm part
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)
        
    def forward(self, x):
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0)
        # linear part forward
        linear_part = torch.sum(self.linear(tmp), dim = 1) + self.bias
        # ffm part forward
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = []
        for i in range(self.num_fields -1):
            for j in range(i+1, self.num_fields):
                ix.append(xs[j][:,i] * xs[i][:,j])
        ix = torch.stack(ix, dim = 1)
        ffm_part = torch.sum(torch.sum(ix, dim=1), dim=1, keepdim=True)
        
        x = linear_part + ffm_part
        x = torch.sigmoid(x.squeeze(1))
        return x
    