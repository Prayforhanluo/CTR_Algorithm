# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:57:48 2021

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn 

class SELayer(nn.Module):
    """
        Squenzee and Excitation Layer in FiBiNET
    """
    
    def __init__(self, field_size, reduction_ratio, pooling = 'mean'):
        super(SELayer, self).__init__()
        self.field_size = field_size
        self.reduction_ratio = reduction_ratio
        self.pooling = pooling
        
        self.reduction_size = max(1, field_size // reduction_ratio)
        
        self.excitation = nn.Sequential(
            nn.Linear(self.field_size, self.reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.reduction_size, self.field_size, bias=False),
            nn.ReLU()
        )
            
    def forward(self, x):
        """
            x : batch * field_size * embed_dim
        """
        # Squeeze
        if self.pooling == 'mean':
            z = x.mean(dim=2)
        elif self.pooling == 'max':
            z = x.max(dim=2)
        else:
            raise Exception('pooling type unknown')
        # Excitation
        A = self.excitation(z)
        
        # Reweight Embedding
        V = x * A.unsqueeze(dim=2)
        return V
    
class BilinearInteraction(nn.Module):
    """
        BilinearInteraction Layer in FiBiNET
    """
    
    def __init__(self, field_size, embed_dim, bilinear_type = 'interaction'):
        super(BilinearInteraction, self).__init__()
        
        self.field_size = field_size
        self.bilinear_type = bilinear_type
        self.bilinear_layer = nn.ModuleList()
        
        if bilinear_type == 'all':
            self.bilinear_layer = nn.Linear(embed_dim, embed_dim, bias = False)
        elif bilinear_type == 'each':
            for _ in range(field_size):
                self.bilinear_layer.append(
                    nn.Linear(embed_dim, embed_dim, bias = False)
                    )
        elif bilinear_type == 'interaction':
            for i in range(field_size):
                for j in range(i+1, field_size):
                    self.bilinear_layer.append(
                        nn.Linear(embed_dim, embed_dim)
                        )
        else:
            raise Exception("bilinear type unknown")
    
    def forward(self, x):
        """
            x : batch * field_size * embed_dim
        """
        
        x = torch.split(x, 1, dim=1) # 将每个field拿出来
        
        p = []
        if self.bilinear_type == 'all':
            for i in range(self.field_size):
                v_i = x[i]
                for j in range(i+1, self.field_size):
                    v_j = x[j]
                    p.append(torch.mul(self.bilinear_layer(v_i), v_j))
        elif self.bilinear_type == 'each':
            for i in range(self.field_size):
                v_i = x[i]
                for j in range(i+1, self.field_size):
                    v_j = x[j]
                    p.append(torch.mul(self.bilinear_layer[i](v_i), v_j))
        elif self.bilinear_type == 'interaction':
            num = 0 #从ModuleList取Vi,Vj对应的W
            for i in range(self.field_size):
                v_i = x[i]
                for j in range(i+1, self.field_size):
                    v_j = x[j]
                    p.append(torch.mul(self.bilinear_layer[num](v_i), v_j))
                    num += 1
        else:
            raise Exception("bilinear type unknown")
        p = torch.cat(p, dim=1)
        return p
                                                

class FiBiNET(nn.Module):
    """
         Feature Importance and Bilinear feature Interaction Net
    """
    
    def __init__(self, feature_fields, embed_dim, reduction_ratio, 
                 pooling = 'mean', mlp_dims = (64, 32), dropout = 0.):
        super(FiBiNET, self).__init__()
        self.field_size = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)        
        # Embedding layer
        self.embedding = nn.Embedding(sum(feature_fields)+1, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        
        # SE layer
        self.SELayer = SELayer(self.field_size, reduction_ratio)

        # Bilinear layer
        self.Bilinear = BilinearInteraction(self.field_size, embed_dim,
                                            bilinear_type='interaction')

        #final DNN layer
        dnn_layers = []
        input_dim = self.field_size * (self.field_size - 1) * embed_dim
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
        
    def forward(self, x):
        """
            x : batch_size * field_size
        """
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0)        
        # embeded dense vector
        embeded_x = self.embedding(tmp)
       
        # SENet-like embeded vec
        SE_embeded_x = self.SELayer(embeded_x)
        
        # Bilinear interaction
        p = self.Bilinear(embeded_x).flatten(start_dim = 1)
        se_p = self.Bilinear(SE_embeded_x).flatten(start_dim = 1)
        
        # final DNN
        concat_p = torch.cat([p, se_p], dim=1)
        res = self.mlp(concat_p)
        
        return torch.sigmoid(res.squeeze(1))
        
        
        
        