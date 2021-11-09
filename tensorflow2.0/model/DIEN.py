# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:29:25 2021

@author: luoh1
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

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

class Dice(layers.Layer):
    """
        Dice Activation Layer
    """
    def __init__(self):
        super(Dice, self).__init__()
        self.epsilon = 1e-9
    
    def build(self, input_shape):
        self.alpha = self.add_weight(name = 'dice_alpha',
                                     shape = (),
                                     initializer = keras.initializers.Zeros(),
                                     trainable = True)
    
    def call(self, x):
        
        top = x - tf.reduce_mean(x, axis=0)
        bottom = tf.sqrt(tf.math.reduce_variance(x, axis=0) + self.epsilon)
        norm_x = top / bottom
        p = tf.sigmoid(norm_x)
        x = self.alpha * x * (1-p) + x * p
        return x


class EmbeddingLayer(Layer):
    """
        Embedding Layer
    """
    def __init__(self, feature_dim, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.embedding = layers.Embedding(feature_dim+1, embed_dim,
                                          mask_zero = True, name = 'item_emb')
    
    def call(self, x, neg_x = None):
        """
            input : 
                x : (behaviors * 40, ads * 1) -> batch * (behaviors + ads)
                neg_x : (behavior * 39) -> batch * behavior
            output :
                query_ad : (batch * 1 * embed_dim) (B * 1 * E)
                user_behavior : (batch * Time_seq_len * embed_dim) (B * T * E)
                behavior_length : (B)
        """
        behaviors_x = x[:,:-1]
        ads_x = x[:,-1]
        
        query_ad = tf.expand_dims(self.embedding(ads_x), axis=1)
        user_behavior = self.embedding(behaviors_x)
        mask = tf.cast(behaviors_x > 0, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        if neg_x is not None:
            neg_mask = tf.cast(neg_x > 0, tf.float32)
            neg_mask = tf.expand_dims(neg_mask, axis=-1)
            neg_user_behavior = self.embedding(neg_x)
            
            return query_ad, user_behavior, mask, \
                   neg_user_behavior, neg_mask
        
        return query_ad, user_behavior, mask


class InterestExtractLayer(Model):
    """
        DIEN中的兴趣抽取层
    """
    def __init__(self, embed_dim, mlp_dims = [100,50], dropout=0.):
        super(InterestExtractLayer, self).__init__()
        
        # 传统的GRU来抽取时序行为的兴趣表示
        
        self.GRU = layers.GRU(embed_dim, return_sequences = True)

        # 用一个mlp来计算 auxiliary loss
        self.auxiliary_mlp = keras.Sequential()
        input_dim = embed_dim*2
        for fc_dim in mlp_dims:
            self.auxiliary_mlp.add(layers.Dense(fc_dim, input_shape = [input_dim,]))
            self.auxiliary_mlp.add(layers.Activation('relu'))
            self.auxiliary_mlp.add(layers.Dropout(dropout))
            self.input_dim = fc_dim
        self.auxiliary_mlp.add(layers.Dense(1, input_shape = [input_dim,]))
        
    
    def call(self, user_behavior, mask, neg_user_behavior = None, neg_mask = None):
        """
            user_behavior : (B, T, E)
            mask : (B, T, 1)
            neg_user_behavior : (B, T-1, E)
            neg_mask : (B, T-1 , 1)
        """
        
        mask_bool = tf.cast(tf.squeeze(mask, axis=2), tf.bool)
        gru_interests = self.GRU(user_behavior, mask = mask_bool) # B * T * E
        
        if neg_user_behavior is not None:
            # 计算Auxiliary Loss
            gru_embed = gru_interests[:,1:]
            # 只在负采样的时候计算 aux loss
            neg_mask_bool = tf.cast(tf.squeeze(neg_mask, axis=2), tf.bool)
            
            # 正样本的构建
            pos_seq = tf.concat([gru_embed, user_behavior[:,1:]], -1) # (B, T-1, 2E)
            pos_res = self.auxiliary_mlp(pos_seq)
            pos_res = tf.sigmoid(pos_res[neg_mask_bool])
            pos_target = tf.ones_like(pos_res, tf.float16)
            
            # 负样本的构建
            neg_seq = tf.concat([gru_embed, neg_user_behavior], -1) # (B, T-1, 2E)
            neg_res = self.auxiliary_mlp(neg_seq)
            neg_res = tf.sigmoid(neg_res[neg_mask_bool])
            neg_target = tf.zeros_like(neg_res, tf.float16)            
            
            aux_loss = keras.losses.binary_crossentropy(tf.concat([pos_res, neg_res],0),
                                                        tf.concat([pos_target, neg_target],0))
            aux_loss = tf.reduce_mean(aux_loss)
            return gru_interests, aux_loss
        
        return gru_interests, 0
    

class ActivationUnit(Model):
    """
        Activation Unit
    """
    def __init__(self, embed_dim, dropout = 0.2, fc_dims = [32, 16]):
        super(ActivationUnit, self).__init__()
        
        self.fc_layers = keras.Sequential()
        input_dim = embed_dim*4
        for fc_dim in fc_dims:
            self.fc_layers.add(layers.Dense(fc_dim, input_shape = [input_dim,]))
            self.fc_layers.add(Dice())
            self.fc_layers.add(layers.Dropout(dropout))
            self.input_dim = fc_dim
        self.fc_layers.add(layers.Dense(1, input_shape = [input_dim,]))
    
    def call(self, query, user_behavior):
        """
            query : 单独的ad的embedding mat -> batch * 1 * embed 
            user_behavior : 行为特征矩阵 -> batch * seq_len * embed            
        """
        
        # repeat ads
        seq_len = user_behavior.shape[1]
        queries = tf.concat([query]*seq_len, axis = 1)
        attn_input = tf.concat([queries,
                                user_behavior,
                                queries - user_behavior,
                                queries * user_behavior], axis = -1)
        out = self.fc_layers(attn_input)
        return out


class AttentionPoolingLayer(Model):
    """
        Attention Pooling Layer
    """
    def __init__(self, embed_dim, dropout, return_score = False):
        super(AttentionPoolingLayer, self).__init__()
        
        self.active_unit = ActivationUnit(embed_dim=embed_dim,dropout=dropout)
        self.return_score = return_score
    
    def call(self, query, user_behavior, mask):
        """
            query_ad : 单独的ad的embedding mat -> batch * 1 * embed
            user_behavior : 行为特征矩阵 -> batch * seq_len * embed
            mask : 被padding为0的行为置为false -> batch * seq_len * 1
        """
        
        # attn weights
        attn_weights = self.active_unit(query, user_behavior)
        # mul weights and sum pooling
        if not self.return_score:
            output = user_behavior * attn_weights * mask
            return output
        
        return attn_weights



class AGRUCell(Layer):
    """
        Attention based GRU (AGRU)
        公式如下:
            r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)
            #z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)
            h' = tanh(W_ih * x + b_ih + r * (W_hh * h + b_hh))
            h = (1 - att_score) * h + att_score * h'

    """
    def __init__(self, units):
        super().__init__()
        self.units = units
        # 作为一个 RNN 的单元，必须有state_size属性
        # state_size 表示每个时间步输出的维度
        self.state_size = units
    
    
    def build(self, input_shape):
        # 输入数据是一个tupe: (gru_embed, atten_scores)
        # 因此，t时刻输入的维度为：
        dim_xt = input_shape[0][-1]
        
        # 重置门中的参数
        self.w_ir = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_ir')
        self.w_hr = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='w_hr')
        self.b_ir = tf.Variable(tf.random.normal(shape=[self.units]), name='b_ir')
        self.b_hr = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hr')
        # 更新门被att_score代替
        
        # 候选隐藏中的参数
        self.w_ih = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_ih')
        self.w_hh = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='w_hh')
        self.b_ih = tf.Variable(tf.random.normal(shape=[self.units]), name='b_ih')
        self.b_hh = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hh')
        
    
    def call(self, inputs, states):
        x_t, att_score = inputs
        states = states[0]
        # 重置门
        r_t = tf.sigmoid(tf.matmul(x_t, self.w_ir) + self.b_ir +\
                         tf.matmul(states, self.w_hr) + self.b_hr)

        # 候选隐藏状态
        h_t_ = tf.tanh(tf.matmul(x_t, self.w_ih) + self.b_ih + \
                       tf.multiply(r_t,(tf.matmul(states, self.w_hh) + self.b_hh)))
        # 输出值
        h_t = tf.multiply(1-att_score, states) + tf.multiply(att_score, h_t_)
        # 对gru而言，当前时刻的output与传递给下一时刻的state相同
        next_state = h_t
        
        return h_t, next_state


class AUGRUCell(Layer):
    """
    
        GRU with attentional update gate (AUGRU)
        公式如下:
            r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)
            z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)
            z = z * att_score
            h' = tanh(W_ih * x + b_ih + r * (W_hh * h + b_hh))
            h = (1 - z) * h + z * h'
            
    """
    def __init__(self, units):
        super().__init__()
        self.units = units
        # 作为一个 RNN 的单元，必须有state_size属性
        # state_size 表示每个时间步输出的维度
        self.state_size = units
    
    
    def build(self, input_shape):
        # 输入数据是一个tupe: (gru_embed, atten_scores)
        # 因此，t时刻输入的维度为：
        dim_xt = input_shape[0][-1]
        
        # 重置门中的参数
        self.w_ir = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_ir')
        self.w_hr = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='w_hr')
        self.b_ir = tf.Variable(tf.random.normal(shape=[self.units]), name='b_ir')
        self.b_hr = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hr')
        
        # 更新门中的参数
        self.w_iz = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_iz')
        self.w_hz = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='W_hz')
        self.b_iz = tf.Variable(tf.random.normal(shape=[self.units]), name='b_iz')
        self.b_hz = tf. Variable(tf.random.normal(shape=[self.units]), name='b_hz')
        
        # 候选隐藏中的参数
        self.w_ih = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_ih')
        self.w_hh = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='w_hh')
        self.b_ih = tf.Variable(tf.random.normal(shape=[self.units]), name='b_ih')
        self.b_hh = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hh')
        
    
    def call(self, inputs, states):
        x_t, att_score = inputs
        states = states[0]
        # 重置门
        r_t = tf.sigmoid(tf.matmul(x_t, self.w_ir) + self.b_ir +\
                         tf.matmul(states, self.w_hr) + self.b_hr)
        # 更新门
        z_t = tf.sigmoid(tf.matmul(x_t, self.w_iz) + self.b_iz +\
                         tf.matmul(states, self.w_hz) + self.b_hz)
        # 带有注意力的更新门
        z_t = tf.multiply(att_score, z_t)
        # 候选隐藏状态
        h_t_ = tf.tanh(tf.matmul(x_t, self.w_ih) + self.b_ih + \
                       tf.multiply(r_t,(tf.matmul(states, self.w_hh) + self.b_hh)))
        # 输出值
        h_t = tf.multiply(1-z_t, states) + tf.multiply(z_t, h_t_)
        # 对gru而言，当前时刻的output与传递给下一时刻的state相同
        next_state = h_t
        
        
        return h_t, next_state


class InterestEvolutionLayer(Model):
    """
        DIEN中的兴趣进化层
    """
    def __init__(self, input_size, gru_type = 'AUGRU', dropout = 0.2):
        super(InterestEvolutionLayer, self).__init__()
        self.gru_type = gru_type
        self.dropout = dropout
        
        if gru_type == 'GRU':
            self.attention = AttentionPoolingLayer(embed_dim = input_size, 
                                                   dropout = dropout)
            self.evolution = layers.GRU(units = input_size, 
                                        return_sequences = True)
        elif gru_type == 'AIGRU':
            self.attention = AttentionPoolingLayer(embed_dim = input_size, 
                                                   dropout = dropout,
                                                   return_score = True)
            self.evolution = layers.GRU(units = input_size)
        elif gru_type == 'AGRU':
            self.attention = AttentionPoolingLayer(embed_dim = input_size, 
                                                   dropout = dropout,
                                                   return_score = True)
            self.evolution = layers.RNN(AGRUCell(units=input_size))
        elif gru_type == 'AUGRU':
            self.attention = AttentionPoolingLayer(embed_dim = input_size, 
                                                   dropout = dropout,
                                                   return_score = True)
            self.evolution = layers.RNN(AUGRUCell(units=input_size))
            
    def call(self, query_ad, gru_interests, mask):
        """
            query_ad : B * 1 * E
            gru_interests : B * T * H
            mask : B * T * 1
        """
        mask_bool = tf.cast(tf.squeeze(mask, axis=2), tf.bool)
        
        if self.gru_type == 'GRU':
            # GRU后接attention
            out = self.evolution(gru_interests, mask = mask_bool)
            out = self.attention(query_ad, out, mask)
            out = tf.reduce_sum(out, axis = 1)   # B * H
        elif self.gru_type == 'AIGRU':
            # AIGRU
            att_score = self.attention(query_ad, gru_interests, mask)
            out = att_score * gru_interests
            out = self.evolution(out, mask = mask_bool)   # B * H
        elif self.gru_type == 'AGRU' or self.gru_type == 'AUGRU':
            # AGRU or AUGRU
            att_score = self.attention(query_ad, gru_interests, mask)
            out = self.evolution((gru_interests, att_score), mask = mask_bool)  # B * H
        
        return out
            
            
class DeepInterestEvolutionNet(Model):
    """
        深度兴趣进化网络
    """
    def __init__(self, feature_dim, embed_dim, mlp_dims, 
                 gru_type = 'AUGRU', dropout = 0.1):
        super(DeepInterestEvolutionNet, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.gru_type = gru_type
        
        # Embedding Layer
        self.embedding = EmbeddingLayer(feature_dim, embed_dim)
        
        # Interest Extract Layer
        self.interest_extract = InterestExtractLayer(embed_dim = embed_dim,
                                                     dropout = dropout)
        
        # Interest Evolution Layer
        self.interest_evolution = InterestEvolutionLayer(input_size=embed_dim,
                                                         gru_type=gru_type,
                                                         dropout=dropout)
        
        # 最后的MLP层预测
        self.final_mlp = keras.Sequential()
        input_dim = embed_dim*2
        for fc_dim in mlp_dims:
            self.final_mlp.add(layers.Dense(fc_dim, input_shape = [input_dim,]))
            self.final_mlp.add(layers.Activation('relu'))
            self.final_mlp.add(layers.Dropout(dropout))
            self.input_dim = fc_dim
        self.final_mlp.add(layers.Dense(1, input_shape = [input_dim,]))
    
    def call(self, x, neg_x = None):
        """
            x : (behaviors * 40, ads * 1) -> batch * (behaviors + ads)
            neg_x : (behaviors * 39) -> batch * (behaviors + ads)
        """
        # Embedding
        _ = self.embedding(x, neg_x)
        if neg_x is not None:
            query_ad, user_behavior, mask, neg_user_behavior, neg_mask = _
        else:
            query_ad, user_behavior, mask = _
            neg_user_behavior = None
            neg_mask = None
        
        # Interest Extraction
        gru_interest, aux_loss = self.interest_extract(user_behavior, 
                                                       mask, 
                                                       neg_user_behavior, 
                                                       neg_mask)
        
        # Interest Evolution
        final_interest = self.interest_evolution(query_ad, gru_interest, mask)
        
        # MLP for prediction
        concat_out = tf.concat([tf.squeeze(query_ad, 1),
                                final_interest], 1)
        out = self.final_mlp(concat_out)
        out = tf.sigmoid(out)
        
        return out, aux_loss
        


