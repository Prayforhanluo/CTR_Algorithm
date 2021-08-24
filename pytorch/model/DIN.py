# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 16:39:44 2021

@author: luoh1
"""

import torch
import torch.nn as nn


class Dice(nn.Module):
    """
        Dice Active Function
    """
    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1,)))
        self.epsilon = 1e-9
    
    def forward(self, x):

        norm_x = (x - x.mean(dim=0)) / torch.sqrt(x.var(dim=0) + self.epsilon)
        p = torch.sigmoid(norm_x)
        x = self.alpha * x.mul(1-p) + x.mul(p)
    
        return x


class ActivationUnit(nn.Module):
    """
         Activatuon Unit
    """
    def __init__(self, embedding_dim, dropout=0.2, fc_dims = [32, 16]):
        super(ActivationUnit, self).__init__()
        
        fc_layers = []
        
        # FC Layers input dim, concat 4 groups
        input_dim = embedding_dim*4     
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(Dice())
            fc_layers.append(nn.Dropout(p = dropout))
            input_dim = fc_dim
        
        fc_layers.append(nn.Linear(input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, query, user_behavior):
        """
            query : 单独的ad的embedding mat -> batch * 1 * embed 
            user_behavior : 行为特征矩阵 -> batch * seq_len * embed
        """
        
        # ads repeat seq_len times
        seq_len = user_behavior.shape[1]
        queries = torch.cat([query] * seq_len, dim=1)
        
        attn_input = torch.cat([queries, user_behavior, 
                                queries - user_behavior, 
                                queries * user_behavior], dim = -1)
        out = self.fc(attn_input)
        return out
    

class AttentionPoolingLayer(nn.Module):
    """
        Attention Sequence Pooling Layer
    """
    def __init__(self, embedding_dim,  dropout):
        super(AttentionPoolingLayer, self).__init__()
        
        self.active_unit = ActivationUnit(embedding_dim = embedding_dim, 
                                          dropout = dropout)
        
    def forward(self, query_ad, user_behavior, mask):
        """
            query_ad : 单独的ad的embedding mat -> batch * 1 * embed
            user_behavior : 行为特征矩阵 -> batch * seq_len * embed
            mask : 被padding为0的行为置为false -> batch * seq_len * 1
        """
        # weights
        attns = self.active_unit(query_ad, user_behavior) # Batch * seq_len * 1        
        # multiply weights and sum pooling 
        output = user_behavior.mul(attns.mul(mask)) # batch * seq_len * embed_dim
        output = user_behavior.sum(dim=1) # batch * embed_dim
        
        return output
    

class DeepInterestNet(nn.Module):
    """
        Deep Interest Network
    """
    def __init__(self, feature_dim, embed_dim, mlp_dims, dropout):
        super(DeepInterestNet, self).__init__()
        self.feature_dim = feature_dim
        self.embedding = nn.Embedding(feature_dim+1, embed_dim)
        self.AttentionActivate = AttentionPoolingLayer(embed_dim, dropout)
                
        # FC Layers for predictions
        fc_layers = []
        # 由于只有用户历史行为和商品本身的ID,这里在embedding后concate后是2个embed size
        input_dim = embed_dim * 2      
        for fc_dim in mlp_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p = dropout))
            input_dim = fc_dim
        
        fc_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*fc_layers)        
    
    def forward(self, x):
        """
            x : (behaviors * 40, ads * 1) -> batch * (behaviors + ads)
        """
        # define mask
        behaviors_x = x[:,:-1]
        mask = (behaviors_x > 0).float().unsqueeze(-1)
        ads_x = x[:,-1]
        
        # embedding
        query_ad = self.embedding(ads_x).unsqueeze(1)
        user_behavior = self.embedding(behaviors_x)
        user_behavior = user_behavior.mul(mask)
        
        # attention pooling layer
        user_interest = self.AttentionActivate(query_ad, user_behavior, mask)
        concat_input = torch.cat([user_interest, query_ad.squeeze(1)], dim = 1)
        
        # MLP prediction
        out = self.mlp(concat_input)
        out = torch.sigmoid(out.squeeze(1))
        
        return out
