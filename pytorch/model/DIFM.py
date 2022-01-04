# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 15:00:25 2021

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionInteract(nn.Module):
    """
        多头注意力的交互层
    """
    
    def __init__(self, embed_size, head_num, dropout, residual = True):
        """
        """
        super(MultiHeadAttentionInteract,self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = embed_size // head_num
        
    
        # self.W_Q = nn.Linear(embed_size, embed_size, bias=False)
        # self.W_K = nn.Linear(embed_size, embed_size, bias=False)
        # self.W_V = nn.Linear(embed_size, embed_size, bias=False)
        
        # 直接定义参数, 更加直观
        self.W_Q = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_K = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_V = nn.Parameter(torch.Tensor(embed_size, embed_size))
        
        if self.use_residual:
            self.W_R = nn.Parameter(torch.Tensor(embed_size, embed_size))
        
        # 初始化, 避免计算得到nan
        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)
        
    
    def forward(self, x):
        """
            x : (batch_size, feature_fields, embed_dim)
        """
        
        # 线性变换到注意力空间中
        Query = torch.tensordot(x, self.W_Q, dims=([-1], [0]))
        Key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        Value = torch.tensordot(x, self.W_V, dims=([-1], [0]))
        
        # Head (head_num, bs, fields, D / head_num)
        Query = torch.stack(torch.split(Query, self.attention_head_size, dim = 2))
        Key = torch.stack(torch.split(Key, self.attention_head_size, dim = 2))
        Value = torch.stack(torch.split(Value, self.attention_head_size, dim = 2))
        
        # 计算内积
        inner = torch.matmul(Query, Key.transpose(-2,-1))
        inner = inner / self.attention_head_size ** 0.5
        
        # Softmax归一化权重
        attn_w = F.softmax(inner, dim=-1)
        attn_w = F.dropout(attn_w, p = self.dropout)
        
        # 加权求和
        results = torch.matmul(attn_w, Value)
        
        # 拼接多头空间
        results = torch.cat(torch.split(results, 1, ), dim = -1)
        results = torch.squeeze(results, dim = 0) # (bs, fields, D)
        
        # 残差学习(resnet YYDS?)
        if self.use_residual:
            results = results + torch.tensordot(x, self.W_R, dims=([-1], [0]))
        
        results = F.relu(results)
        
        return results
    

class FMLayer(nn.Module):
    """
        FM layer
    """
    
    def __init__(self):
        super(FMLayer, self).__init__()
    
    def forward(self, x):
        """
            x : batch * field_size * embed_dim
        """
        tmp = x
        square_of_sum = torch.sum(tmp, dim=1) ** 2
        sum_of_square = torch.sum(tmp ** 2, dim=1)
        cross_part = square_of_sum - sum_of_square
        cross_part = 0.5 * torch.sum(cross_part, dim=1, keepdim=True)
        
        return cross_part


class BitWiseFEN(nn.Module):
    """
        Bit Wise Net Layer in DIFM
    """
    def __init__(self, input_dim, mlp_dims = [128, 64], dropout = 0.0):
        super(BitWiseFEN, self).__init__()
        
        dnn_layers = []
        self.mlp_dims = mlp_dims
        for mlp_dim in mlp_dims:
            # 全连接层
            dnn_layers.append(nn.Linear(input_dim, mlp_dim))
            dnn_layers.append(nn.BatchNorm1d(mlp_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(p = dropout))
            input_dim = mlp_dim
        self.mlp = nn.Sequential(*dnn_layers)
        
    def forward(self, x):
        """
            x : batch, field * embed_dim
        """
        
        return self.mlp(x)


class FENLayer(nn.Module):
    """
        Dual-FEN Layer in DIFM
    """
    def __init__(self, field_dim, embed_size, head_num, mlp_dims = [128,64], dropout = 0.0):
        super(FENLayer, self).__init__()
        
        self.bit_wise_net = BitWiseFEN(input_dim = field_dim * embed_size,
                                       mlp_dims = mlp_dims,
                                       dropout = dropout)
        
        self.vec_wise_net = MultiHeadAttentionInteract(embed_size = embed_size, 
                                                       head_num = head_num, 
                                                       dropout = dropout)
        
        
        # Combination Layer
        # h field feature -> h weights
        self.trans_bit_nn = nn.Linear(in_features = mlp_dims[-1], 
                                      out_features = field_dim)
        
        self.trans_vec_nn = nn.Linear(in_features = field_dim * embed_size, 
                                      out_features = field_dim)
        
    
    def forward(self, x):
        """
            x : batch, field_dim, embed_dim
        """
        b, f, e = x.shape
        bit_wise_x = self.bit_wise_net(x.reshape(b, f * e))
        vec_wise_x = self.vec_wise_net(x).reshape(b, f * e)
        
        m_bit = self.trans_bit_nn(bit_wise_x)
        m_vec = self.trans_vec_nn(vec_wise_x)
        
        m_x = m_vec + m_bit
        
        return m_x


class DIFM(nn.Module):
    """
        A Dual Input-aware Factorization Machine for CTR Prediction
    """
    def __init__(self, feature_fields, embed_size, head_num, mlp_dims = [128, 64],
                 dropout = 0.0):
        super(DIFM, self).__init__()
        self.field_dim = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        
        #embedding 中的 weights作为ReweightingLayer的第一部分
        self.sparse_weight = torch.nn.Embedding(sum(feature_fields), 1)        
        
        #embedding layer
        self.embedding = torch.nn.Embedding(sum(feature_fields), embed_size)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        
        #FEN layer
        self.FEN = FENLayer(self.field_dim, embed_size, head_num, mlp_dims=mlp_dims)
        
        self.FM = FMLayer()
        
    def forward(self, x):
        """
        """
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0)
        # embedding weights
        sparse_weights = self.sparse_weight(tmp)    
        # embedding
        embed_x = self.embedding(tmp)  
        # m_x
        m_x = self.FEN(embed_x)
        
        # Reweighting Layer
        # w * x
        sparse_weights = sparse_weights.squeeze() * m_x
        sparse_weights = sparse_weights.sum(dim = 1, keepdim=True)    
        # <vxi, vxj>(xi,xj)
        fm_input = embed_x * m_x.unsqueeze(-1)
        fm_out = self.FM(fm_input)
        
        # Predicting
        logit = sparse_weights + fm_out
        
        return torch.sigmoid(logit).squeeze(1)
        

