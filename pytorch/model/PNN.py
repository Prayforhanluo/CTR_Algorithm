# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 11:37:32 2021

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn


class InnerProductNetwork(nn.Module):

    def forward(self, x):
        """
        x : (batch_size, num_fields, embed_dim)
        """
        num_fields = x.shape[1]
        row, col = [],[]
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)


class OuterProductNetwork(nn.Module):

    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        x : (batch_size, num_fields, embed_dim)
        """
        num_fields = x.shape[1]
        row, col = [],[]
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        if self.kernel_type == 'mat':
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)
            return torch.sum(kp * q, -1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)
    
    
class PNN(nn.Module):
    """
        Product-based Neural Network
    """
    def __init__(self, feature_fields, embed_dim, mlp_dims, dropout, method = 'inner'):
        super(PNN, self).__init__()
        self.feature_fields = feature_fields
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        if method not in ['inner', 'outer']:
            raise ValueError ('unknown product type : %s' % method)
        else:
            self.method = method
        
        # Embedding layer
        self.embedding = nn.Embedding(sum(feature_fields)+1, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        self.embedding_out_dim = len(feature_fields) * embed_dim
        
        #DNN layer
        dnn_layers = []
        num_fields = len(feature_fields)
        input_dim = self.embedding_out_dim + (num_fields * (num_fields - 1))// 2 
        self.mlp_dims = mlp_dims
        for mlp_dim in mlp_dims:
            # 全连接层
            dnn_layers.append(nn.Linear(input_dim, mlp_dim))
            dnn_layers.append(nn.BatchNorm1d(mlp_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(p = dropout))
            input_dim = mlp_dim
        dnn_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*dnn_layers) 
        
        #Product layer
        if self.method == 'inner':
            self.pn = InnerProductNetwork()
        else:
            self.pn = OuterProductNetwork(num_fields, embed_dim)
    
    def forward(self, x):
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0)
        
        # embeded dense vector
        embeded_x = self.embedding(tmp)       
        # product connection
        product_part = self.pn(embeded_x)
        x = torch.cat([embeded_x.view(-1, self.embedding_out_dim), product_part], dim = 1)
        x = self.mlp(x)
        x = torch.sigmoid(x.squeeze(1))
        return x
    
    