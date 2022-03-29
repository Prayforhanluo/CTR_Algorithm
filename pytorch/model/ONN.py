# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:41:49 2022

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn


class ONN(nn.Module):
    """
        Operation-aware Neural Networks
        FFM & PNN = ONN
        刚开始读ONN论文，感觉就像灌水模型，FFM 跟 PNN 拼接成了 ONN
        这个模型的参数量远超其他
        在embedding的时候，每个field的embedding在做不同操作的时候，都会不一样
        个人觉得简单来说就是粗暴的拓宽了假设空间
        这个模型给我的最大的启示是：
        O(n2) > O(n)
        简单粗暴的增加模型复杂度的确有好处
        缺点就是慢！！！
    """
    def __init__(self, feature_fields, embed_dim, mlp_dims, dropout):
        super(ONN, self).__init__()
        self.feature_fields = feature_fields
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype=np.long)
        self.num_fields = len(feature_fields)
        
        # embeddings
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(feature_fields), embed_dim) for _ in range(self.num_fields)
        ])
        

        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)
            
        #DNN layer
        dnn_layers = []
        num_fields = len(feature_fields)
        input_dim = embed_dim * num_fields + (num_fields * (num_fields - 1))// 2 
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
        """
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0)
        # embedding matrix
        field_aware_embeds = [embedding(tmp) for embedding in self.embeddings]
        # raw embedding
        raw_embed = field_aware_embeds[-1].flatten(start_dim=1)
        # Field-aware Inner product
        interaction = []
        for i in range(self.num_fields - 1):
            for j in range(i+1, self.num_fields):
                tmp_i_j = field_aware_embeds[j-1][:, i, :]
                tmp_j_i = field_aware_embeds[i][:, j, :]
                tmp_dot = torch.sum(tmp_i_j * tmp_j_i, dim = 1, keepdim=True)
                interaction.append(tmp_dot)
            
        ffm_out = torch.cat(interaction, dim = 1)
        # DNN 
        dnn_input = torch.cat([raw_embed, ffm_out], dim = 1)
        dnn_part = self.mlp(dnn_input)
        out = torch.sigmoid(dnn_part.squeeze(1))
        
        return out
    
        
        
