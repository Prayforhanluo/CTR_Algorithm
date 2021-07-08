# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:17:10 2021

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn

class DeepCrossNet(nn.Module):
    """
        Deep Cross Network
    """
    def __init__(self, feature_fields, embed_dim, num_layers, mlp_dims, dropout):
        """
        """
        super(DeepCrossNet, self).__init__()
        self.feature_fields = feature_fields
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        
        # Embedding layer
        self.embedding = nn.Embedding(sum(feature_fields)+1, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        self.embedding_out_dim = len(feature_fields) * embed_dim
        
        #DNN layer
        dnn_layers = []
        input_dim = self.embedding_out_dim
        self.mlp_dims = mlp_dims
        for mlp_dim in mlp_dims:
            # 全连接层
            dnn_layers.append(nn.Linear(input_dim, mlp_dim))
            dnn_layers.append(nn.BatchNorm1d(mlp_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(p = dropout))
            input_dim = mlp_dim
        self.mlp = nn.Sequential(*dnn_layers)      
    
        # Corss Net layer
        self.num_layers = num_layers
        self.cross_w = nn.ModuleList([
                nn.Linear(self.embedding_out_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.cross_b = nn.ParameterList([
                nn.Parameter(torch.zeros((self.embedding_out_dim,))) for _ in range(num_layers)
        ])
        
        # LR layer
        self.lr = nn.Linear(self.mlp_dims[-1]+self.embedding_out_dim, 1)
        
        
    def forward(self, x):
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0)
        
        # embeded dense vector
        embeded_x = self.embedding(tmp).view(-1, self.embedding_out_dim)
        # DNN out
        mlp_part = self.mlp(embeded_x)
        # Cross Net out
        x0 = embeded_x
        cross = embeded_x
        for i in range(self.num_layers):
            xw = self.cross_w[i](cross)
            cross = x0 * xw + self.cross_b[i] + cross
        
        # stack output
        out = torch.cat([cross, mlp_part], dim = 1)
        
        # LR out
        out = self.lr(out)
        out = torch.sigmoid(out.squeeze(1))
        
        return out
