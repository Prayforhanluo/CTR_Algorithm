# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:30:35 2021

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn

class CrossNetMatrix(nn.Module):
    """
        CrossNet of DCN-v2
    """

    def __init__(self, in_features, layer_num=2):
        super(CrossNetMatrix, self).__init__()
        self.layer_num = layer_num
        # Cross中的W参数 (layer_num,  [W])
        self.weights = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        # Cross中的b参数 (layer_num, [B])
        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        # Init
        for i in range(self.layer_num):
            nn.init.xavier_normal_(self.weights[i])
        for i in range(self.layer_num):
            nn.init.zeros_(self.bias[i])

    def forward(self, x):
        """
            x : batch_size  *  in_features
        """
        x0 = x.unsqueeze(2)
        xl = x.unsqueeze(2)
        for i in range(self.layer_num):
            tmp = torch.matmul(self.weights[i], xl) + self.bias[i]
            xl = x0 * tmp + xl
        xl = xl.squeeze(2)
        
        return xl
    
    
class CrossNetMix(nn.Module):
    """
        CrossNet of DCN-V2 with Mixture of Low-rank Experts
        公式如下：
            G_i(xl) = Linear(xl)
            E_i(xl) = x0·(Ul*g(Cl*g(Vl*xl)) + bl)
            g() = tanh activate func
    """

    def __init__(self, in_features, low_rank = 16, expert_num = 4, layer_num=2):
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.expert_num = expert_num
        
        # Cross中的U参数(layer_num, expert_num, in_features, low_rank)
        self.U_params = nn.Parameter(torch.Tensor(layer_num, expert_num, in_features, low_rank))
        # Cross中的V^T参数(layer_num, expert_num, low_rank, in_features)
        self.V_params = nn.Parameter(torch.Tensor(layer_num, expert_num, low_rank, in_features))
        # Cross中的C参数(layer_num, expert_num, low_rank, low_rank)
        self.C_params = nn.Parameter(torch.Tensor(layer_num, expert_num, low_rank, low_rank))
        # Cross中的bias(layer_num, in_features, 1)
        self.bias = nn.Parameter(torch.Tensor(layer_num, in_features, 1))
        
        # MOE 中的门控gate
        self.gates = nn.ModuleList([nn.Linear(in_features, 1, bias=False) for i in range(expert_num)])
        
        # Init
        for i in range(self.layer_num):
            nn.init.xavier_normal_(self.U_params[i])
            nn.init.xavier_normal_(self.V_params[i])
            nn.init.xavier_normal_(self.C_params[i])
        for i in range(self.layer_num):
            nn.init.zeros_(self.bias[i])
            
    def forward(self, x):
        """
            x : batch_size  *  in_features
        """
        x0 = x.unsqueeze(2)
        xl = x.unsqueeze(2)
        for i in range(self.layer_num):
            expert_outputs = []
            gate_scores = []
            for expert in range(self.expert_num):
                # gate score : G(xl)
                gate_scores.append(self.gates[expert](xl.squeeze(2)))
        
                # cross part
                # g(Vl·xl))
                tmp = torch.tanh(torch.matmul(self.V_params[i][expert], xl))
                # g(Cl·g(Vl·xl))
                tmp = torch.tanh(torch.matmul(self.C_params[i][expert], tmp))
                # Ul·g(Cl·g(Vl·xl)) + bl
                tmp =  torch.matmul(self.U_params[i][expert], tmp) + self.bias[i]
                # E_i(xl) = x0·(Ul·g(Cl·g(Vl·xl)) + bl)
                tmp = x0 * tmp                
                expert_outputs.append(tmp.squeeze(2))
            
            expert_outputs = torch.stack(expert_outputs, 2) # batch * in_features * expert_num
            gate_scores = torch.stack(gate_scores, 1) # batch * expert_num * 1
            MOE_out = torch.matmul(expert_outputs, gate_scores.softmax(1))
            xl = MOE_out + xl  # batch * in_features * 1
        
        xl = xl.squeeze(2)
        
        return xl


class DeepCrossNetv2(nn.Module):
    """
        Deep Cross Network V2
    """
    def __init__(self, feature_fields, embed_dim, layer_num, mlp_dims, dropout = 0.1,
                 cross_method = 'Mix', model_method = 'parallel'):
        """
        """
        super(DeepCrossNetv2, self).__init__()
        self.feature_fields = feature_fields
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        self.model_method = model_method
        
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
        
        if cross_method == 'Mix':
            self.CrossNet = CrossNetMix(in_features=self.embedding_out_dim)
        elif cross_method == 'Matrix':
            self.CrossNet = CrossNetMatrix(in_features=self.embedding_out_dim)
        else:
            raise NotImplementedError
    
        # predict layer
        if self.model_method == 'parallel':
            self.fc = nn.Linear(self.mlp_dims[-1]+self.embedding_out_dim, 1)
        elif self.model_method == 'stack':
            self.fc = nn.Linear(self.mlp_dims[-1], 1)
        else:
            raise NotImplementedError
        
    def forward(self, x):
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0)
        
        # embeded dense vector
        embeded_x = self.embedding(tmp).view(-1, self.embedding_out_dim)
        if self.model_method == 'parallel':
            # DNN out
            mlp_part = self.mlp(embeded_x)
            # Cross part
            cross = self.CrossNet(embeded_x)
            # stack output
            out = torch.cat([cross, mlp_part], dim = 1)
        elif self.model_method == 'stack':
            # Cross part
            cross = self.CrossNet(embeded_x)
            # DNN out
            out = self.mlp(cross)
        # predict out
        out = self.fc(out)
        out = torch.sigmoid(out.squeeze(1))
        
        return out
    
            
                