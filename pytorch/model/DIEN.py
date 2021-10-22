# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 12:13:08 2021

@author: luoh1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def auxiliary_sample(data):
    """
    在DIEN算法中，兴趣抽取层在用GRU来抽取interest的同时，为了使得抽取的interest的表示更加
    合理，该层设计了一个二分类的模型来计算兴趣抽取的准确性，用用户的下一时刻真实的行为作为positive
    sample, 负采样得到的行为作为negative sample来计算一个辅助的loss
    这里是负采样生成辅助的样本。

    Parameters
    ----------
    data : pandas-df
        Encoded data(pandas dataframe)

    Returns
    -------
    neg_sample : numpy.ndarray
        negative samples

    """
    
    cate_max = np.max(data.iloc[:,:-1].values) # 获取所有行为的code
    pos_sample = data.iloc[:, 1:-1].values  #去掉最后的cateID和无下一时刻的第一列
    neg_sample = np.zeros_like(pos_sample)
    
    for i in range(pos_sample.shape[0]):
        for j in range(pos_sample.shape[1]):
            if pos_sample[i, j] > 0:
                idx = np.random.randint(low=1, high = cate_max+1)
                while idx == pos_sample[i,j]:
                    idx = np.random.randint(low=1, high = cate_max+1)
                neg_sample[i, j] = idx
            else:
                break # 后面的行为都是padding的0
    return neg_sample


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

class EmbeddingLayer(nn.Module):
    """
        相比与之前的模型的实现方式, 单独将embedding领出来
    """
    
    def __init__(self, feature_dim, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(feature_dim+1, embed_dim)
    
    def forward(self, x, neg_x = None):
        """
            input : 
                x : (behaviors * 40, ads * 1) -> batch * (behaviors + ads)
                neg_x : (behavior * 39) -> batch * behavior
            output :
                query_ad : (batch * 1 * embed_dim) (B * 1 * E)
                user_behavior : (batch * Time_seq_len * embed_dim) (B * T * E)
                behavior_length : (B)
        """
        # 得到embedding表示, 以及pack_paded_sequence的输入
        behaviors_x = x[:,:-1]
        mask = (behaviors_x > 0).float().unsqueeze(-1)
        ads_x = x[:,-1]
        
        # embedding
        query_ad = self.embedding(ads_x).unsqueeze(1)
        user_behavior = self.embedding(behaviors_x)
        user_behavior = user_behavior.mul(mask)
        behavior_length = (x[:, :-1] > 0).sum(dim=1).cpu()
        if neg_x is not None:
            neg_mask = (neg_x > 0).float().unsqueeze(-1)
            neg_user_behavior = self.embedding(neg_x)
            neg_user_behavior = neg_user_behavior.mul(neg_mask)
        else:
            neg_user_behavior = None
        
        return query_ad, user_behavior, behavior_length, neg_user_behavior


class InterestExtractLayer(nn.Module):
    """
        DIEN中的兴趣抽取层
    """
    def __init__(self, embed_dim, hidden_size, init_std = 0.001, dropout = 0):
        super(InterestExtractLayer, self).__init__()
        
        # 传统的GRU来抽取时序行为的兴趣
        self.GRU = DynamicGRU(embed_dim, hidden_size, gru_type='GRU')
        for name, tensor in self.GRU.named_parameters():
            nn.init.normal_(tensor, mean=0, std = init_std)
        
        # 相比于论文的点积, 代码中用一个[100,50,1]的全连接层来计算auxiliary_loss
        auxiliary_mlp = []
        input_dim = embed_dim * 2      
        for fc_dim in [100, 50]:
            auxiliary_mlp.append(nn.Linear(input_dim, fc_dim))
            auxiliary_mlp.append(nn.ReLU())
            auxiliary_mlp.append(nn.Dropout(p = dropout))
            input_dim = fc_dim
        
        auxiliary_mlp.append(nn.Linear(input_dim, 1))
        auxiliary_mlp.append(nn.Sigmoid())
        self.auxiliary_mlp = nn.Sequential(*auxiliary_mlp)         
    
    def forward(self, user_behavior, behavior_length, neg_user_behavior = None):
        """
            user_behavior : (B * T * E)
            behavior_length : (B,)
            neg_user_behavior : (B * T-1 * E)
        """
        
        max_seq_len = user_behavior.shape[1]
        inputs = pack_padded_sequence(user_behavior, behavior_length, 
                                      batch_first=True, enforce_sorted=False)
        
        packed_interests, hn = self.GRU(inputs)
        
        # pad_interests : (B * T * E) 还原形状
        pad_interests, _ = pad_packed_sequence(packed_interests, batch_first=True,
                                            padding_value=0., total_length=max_seq_len)
        
        if neg_user_behavior is not None:
            # 计算辅助loss
            gru_embed = pad_interests[:,1:]  # 去掉最新的时刻
            batch_size, aux_seq_len, embed_dim = gru_embed.shape
            aux_length = behavior_length - 1 # 计算辅助loss的未填充的序列长度
            # mask作用 : 只在负采样的时候计算aux loss
            mask = (torch.arange(aux_seq_len).repeat(batch_size,1)<aux_length.view(-1,1)).float()
            
            # 正样本的构建,过全连接层
            pos_seq = torch.cat([gru_embed, user_behavior[:,1:]], dim = -1) # (B, T, 2E)
            pos_seq = pos_seq.view(batch_size * aux_seq_len, embed_dim*2)   # (B*T, 2E) 
            pos_res = self.auxiliary_mlp(pos_seq)
            pos_res = pos_res.view(batch_size, aux_seq_len)[mask > 0].view(-1,1)
            pos_target = torch.ones(pos_res.size(), dtype = torch.float)
            
            # 负样本的构建,过全连接层
            neg_seq = torch.cat([gru_embed, neg_user_behavior], dim = -1)  # (B, T, 2E)
            neg_seq = neg_seq.view(batch_size * aux_seq_len, embed_dim*2)  # (B*T, 2E) 
            neg_res = self.auxiliary_mlp(neg_seq)
            neg_res = neg_res.view(batch_size, aux_seq_len)[mask > 0].view(-1,1)
            neg_target = torch.zeros(neg_res.size(), dtype = torch.float)
            
            auxiliary_loss = F.binary_cross_entropy(torch.cat([pos_res,neg_res],dim=0),
                                                    torch.cat([pos_target,neg_target],dim=0))
            return pad_interests, auxiliary_loss
        
        return pad_interests, 0

    
class ActivationUnit(nn.Module):

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

    def __init__(self, embedding_dim,  dropout, return_score = False):
        super(AttentionPoolingLayer, self).__init__()
        
        self.active_unit = ActivationUnit(embedding_dim = embedding_dim, 
                                          dropout = dropout)
        
        self.return_score = return_score
        
    def forward(self, query_ad, user_behavior):
        """
            query_ad : 单独的ad的embedding mat -> batch * 1 * embed
            user_behavior : 行为特征矩阵 -> batch * seq_len * embed
        """
        # weights
        attns = self.active_unit(query_ad, user_behavior) # Batch * seq_len * 1        
        # multiply weights and sum pooling
        if not self.return_score: 
            output = user_behavior.mul(attns) # batch * seq_len * embed_dim
            return output
        return attns
       #output = user_behavior.sum(dim=1) # batch * embed_dim
        
       #return output

class AGRUCell(nn.Module):
    """
        Attention based GRU (AGRU)
        公式如下:
            r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)
            z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)
            h' = tanh(W_ih * x + b_ih + r * (W_hh * h + b_hh))
            h = (1 - att_score) * h + att_score * h'

    """
    def __init__(self, input_size, hidden_size, bias = True):
        super(AGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # W_ir | W_iz | W_ih
        self.weight_ih = nn.Parameter(torch.Tensor(3*hidden_size, input_size))
        # W_hr | W_hz | W_hh
        self.weight_hh = nn.Parameter(torch.Tensor(3*hidden_size, hidden_size))
        self.register_parameter('weight_ih', self.weight_ih)
        self.register_parameter('weight_hh', self.weight_hh)
        
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            self.register_parameter('bias_hh', self.bias_hh)
            for bias_tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(bias_tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
    def forward(self, inputs, hidden_state, att_score):
        # 计算 Wx + b
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hidden_state, self.weight_hh, self.bias_hh)
        
        # 得到每个Wx + b部分
        # i_r : W_ir * x + b_ir
        # i_z : W_iz * x + b_iz
        # i_h : W_ih * x + b_ih
        i_r, i_z, i_h = gi.chunk(3, 1)
        
        # h_r : W_hr * h + b_hr
        # h_z : W_hz * h + b_hz
        # h_h : W_hh * h + b_hh
        h_r, h_z, h_h = gh.chunk(3, 1)
        
        # 计算r, z, h', h
        reset_gate = torch.sigmoid(i_r + h_r) # r
        # update_gate = torch.sigmoid(i_z + h_z) # z
        new_hidden = torch.tanh(i_h + reset_gate * h_h) # h'
        #att_score = att_score.view(-1,1)
        new_hidden = (1 - att_score) * hidden_state + att_score * new_hidden # h
        
        return new_hidden
        
        
class AUGRUCell(nn.Module):
    """
        GRU with attentional update gate (AUGRU)
        公式如下:
            r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)
            z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)
            z = z * att_score
            h' = tanh(W_ih * x + b_ih + r * (W_hh * h + b_hh))
            h = (1 - z) * h + z * h'

    """
    def __init__(self, input_size, hidden_size, bias = True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # W_ir | W_iz | W_ih
        self.weight_ih = nn.Parameter(torch.Tensor(3*hidden_size, input_size))
        # W_hr | W_hz | W_hh
        self.weight_hh = nn.Parameter(torch.Tensor(3*hidden_size, hidden_size))
        self.register_parameter('weight_ih', self.weight_ih)
        self.register_parameter('weight_hh', self.weight_hh)
        
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            self.register_parameter('bias_hh', self.bias_hh)
            for bias_tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(bias_tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
    def forward(self, inputs, hidden_state, att_score):
        # 计算 Wx + b
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hidden_state, self.weight_hh, self.bias_hh)
        
        # 得到每个Wx + b部分
        # i_r : W_ir * x + b_ir
        # i_z : W_iz * x + b_iz
        # i_h : W_ih * x + b_ih
        i_r, i_z, i_h = gi.chunk(3, 1)
        
        # h_r : W_hr * h + b_hr
        # h_z : W_hz * h + b_hz
        # h_h : W_hh * h + b_hh
        h_r, h_z, h_h = gh.chunk(3, 1)
        
        # 计算r, z, h', h
        reset_gate = torch.sigmoid(i_r + h_r) # r
        update_gate = torch.sigmoid(i_z + h_z) # z
        new_hidden = torch.tanh(i_h + reset_gate * h_h) # h'
        # att_score = att_score.view(-1,1)
        update_gate = att_score * update_gate
        new_hidden = (1 - att_score) * hidden_state + att_score * new_hidden # h
        
        return new_hidden    

    
class DynamicGRU(nn.Module):
    """
        实现常规GRU以及DIEN中的几种变种GRU
    """
    def __init__(self, input_size, hidden_size, bias = True,
                 gru_type = 'AUGRU'):
        super(DynamicGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_type = gru_type
        
        if gru_type == 'GRU':
            self.gru_cell = nn.GRUCell(input_size, hidden_size)
        elif gru_type == 'AGRU':
            self.gru_cell = AGRUCell(input_size, hidden_size, bias)
        elif gru_type == 'AUGRU':
            self.gru_cell = AUGRUCell(input_size, hidden_size, bias)
    
    def forward(self, inputs, att_score = None, hidden_state = None):
        """
        input:
            inputs : packed_sequence
            att_score : None if GRU else attention scores
            hidden_state : None for init
        
        output:
            packed_sequence
            
        """
        x, batch_sizes, sorted_indices, unsorted_indices = inputs
        
        if hidden_state is None:
            h = torch.zeros(batch_sizes[0], self.hidden_size, dtype=x.dtype)
            
        output = torch.zeros(x.shape[0], self.hidden_size)
        output_h = torch.zeros(batch_sizes[0], self.hidden_size)
        
        start = 0
        for batch in batch_sizes:
            x_ = x[start:start+batch]
            h_ = h[:batch]
            if self.gru_type == 'GRU':
                h = self.gru_cell(x_, h_)
            else:
                att_ = att_score[start:start+batch]
                h = self.gru_cell(x_, h_, att_)
            output_h[:batch] = h
            output[start:start+batch] = h
            start += batch
        
        return PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices),\
               output_h[unsorted_indices]


class InterestEvolutionLayer(nn.Module):
    """
        DIEN兴趣进化层
    """
    def __init__(self, input_size, gru_type = 'AUGRU', init_std = 0.001, dropout=0.2):
        super(InterestEvolutionLayer, self).__init__()
        self.gru_type = gru_type
        self.init_std = init_std
        self.dropout = dropout
        
        if gru_type == 'GRU':
            self.attention = AttentionPoolingLayer(embedding_dim=input_size, 
                                                   dropout = dropout)
            self.evolution = DynamicGRU(input_size=input_size, 
                                        hidden_size=input_size, 
                                        gru_type='GRU')
        elif gru_type == 'AIGRU':
            self.attention = AttentionPoolingLayer(embedding_dim=input_size, 
                                                   dropout=dropout, 
                                                   return_score=True)
            self.evolution = DynamicGRU(input_size=input_size, 
                                        hidden_size=input_size, 
                                        gru_type='GRU')
        elif gru_type == 'AGRU' or gru_type == 'AUGRU':
            self.attention = AttentionPoolingLayer(embedding_dim=input_size, 
                                                   dropout=dropout, 
                                                   return_score=True)
            self.evolution = DynamicGRU(input_size=input_size, 
                                        hidden_size=input_size,
                                        gru_type=gru_type)
        for name, tensor in self.evolution.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
            
    def forward(self, query_ad, pad_interests, behavior_length):
        """
            query_ad : B * 1 * E
            pad_interests : B * T * E
            behavior_length : (B,)
        """
        max_seq_len = pad_interests.shape[1]
        
        if self.gru_type == 'GRU':
            # GRU 接 Attention
            packed_interest = pack_padded_sequence(pad_interests, behavior_length, 
                                                   batch_first=True, enforce_sorted=False)
            out, _ = self.evolution(packed_interest)
            out, _ = pad_packed_sequence(out, batch_first=True,
                                         total_length=max_seq_len) # B * T * H
            out = self.attention(query_ad, out)
            out = out.sum(dim=1) # B * H
            
        elif self.gru_type == 'AIGRU':
            # AIGRU
            att_score = self.attention(query_ad, pad_interests)
            pad_interests = att_score * pad_interests
            packed_interest = pack_padded_sequence(pad_interests, behavior_length, 
                                                   batch_first=True, enforce_sorted=False)
            out, _ = self.evolution(packed_interest)
            out, _ = pad_packed_sequence(out, batch_first=True,
                                         total_length=max_seq_len) # B * T * H            
            out = out.sum(dim=1) # B * H
            
        elif self.gru_type == 'AGRU' or self.gru_type == 'AUGRU':
            # AGRU or AUGRU
            att_score = self.attention(query_ad, pad_interests)
            packed_att_score= pack_padded_sequence(att_score, behavior_length, 
                                                   batch_first=True, enforce_sorted=False)
            packed_interest = pack_padded_sequence(pad_interests, behavior_length, 
                                                   batch_first=True, enforce_sorted=False)
            out, _ = self.evolution(packed_interest, att_score = packed_att_score.data)
            out, _ = pad_packed_sequence(out, batch_first=True,
                                         total_length=max_seq_len) # B * T * H   
            
            out = out.sum(dim = 1) # B * H
        
        return out
            
        
class DeepInterestEvolutionNet(nn.Module):
    """
        深度兴趣进化网络
    """
    def __init__(self, feature_dim, embed_dim, hidden_size, mlp_dims, 
                 gru_type = 'AUGRU', dropout=0.2):
        super(DeepInterestEvolutionNet, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.gru_type = gru_type
        
        # Embedding layer
        self.embed_layer = EmbeddingLayer(feature_dim, embed_dim)
        
        # Interest Extract Layer
        self.interest_extract = InterestExtractLayer(embed_dim, hidden_size)
        
        # Interest Evolution Layer
        self.interet_evolution = InterestEvolutionLayer(input_size=hidden_size,
                                                        gru_type=gru_type,
                                                        dropout=dropout)
        # Final DNN
        fc_layers = []
        input_dim = embed_dim + hidden_size      
        for fc_dim in mlp_dims:
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p = dropout))
            input_dim = fc_dim
        
        fc_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*fc_layers)
    
    def forward(self, x, neg_x = None):
        """
            x : (behaviors * 40, ads * 1) -> batch * (behaviors + ads)
            neg_x : (behaviors * 39) -> batch * (behaviors + ads)
        """
        
        # Embedding
        query_ad, user_behavior, behavior_length, neg_user_behavior = self.embed_layer(x, neg_x)
        
        # Extracting Interest
        pad_intersts, auxiliary_loss = self.interest_extract(user_behavior,
                                                             behavior_length,
                                                             neg_user_behavior)
        
        # Interest Evolution
        pad_ev_interest = self.interet_evolution(query_ad, 
                                                 pad_intersts,
                                                 behavior_length)
        # DNN for predicting
        concat_input = torch.cat([pad_ev_interest, query_ad.squeeze(1)], dim = 1)
        out = self.mlp(concat_input)
        out = torch.sigmoid(out.squeeze(1))
               
        return out, auxiliary_loss
        
   
        