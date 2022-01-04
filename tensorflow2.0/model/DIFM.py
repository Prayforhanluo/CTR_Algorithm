# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 14:07:54 2022

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

class MultiHeadAttentionInteract(Model):
    """
        多头注意力的交互层
    """
    def __init__(self, embed_size, head_num, dropout, residual = True):
        """
        """
        super(MultiHeadAttentionInteract, self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = embed_size // head_num
    
    def build(self, input_shape):
        """
            定义参数
        """
        
        self.W_Q = self.add_weight(name = 'Query_weight',
                                   shape = (self.embed_size, self.embed_size),
                                   initializer = 'random_normal',
                                   trainable = True)
        

        self.W_K = self.add_weight(name = 'Key_weight',
                                   shape = (self.embed_size, self.embed_size),
                                   initializer = 'random_normal',
                                   trainable = True)
        
        self.W_V = self.add_weight(name = 'Value_weight',
                                   shape = (self.embed_size, self.embed_size),
                                   initializer = 'random_normal',
                                   trainable = True)
        if self.use_residual:
            self.W_R = self.add_weight(name = 'residual_weight',
                                       shape = (self.embed_size, self.embed_size),
                                       initializer = 'random_normal',
                                       trainable = True)
    
    def call(self, x):
        """
            x : (batch_size, feature_fields, embed_dim)
        """
        
        # 线性变换得到Q,K,V
        Query = tf.tensordot(x, self.W_Q, axes = (-1,0))
        Key = tf.tensordot(x, self.W_K, axes = (-1,0))
        Value = tf.tensordot(x, self.W_V, axes = (-1,0))
        
        # Head (head_num, bs, fields, D / head_num)
        Query = tf.stack(tf.split(Query, self.attention_head_size, axis = 2))
        Key = tf.stack(tf.split(Key, self.attention_head_size, axis = 2))
        Value = tf.stack(tf.split(Value, self.attention_head_size, axis = 2))
        
        # Inner Product
        inner = tf.matmul(Query, Key, transpose_b = True)
        inner = inner / self.attention_head_size ** 0.5
        
        # Softmax
        attn_w = tf.nn.softmax(inner, axis = -1)
        attn_w = tf.nn.dropout(inner, rate = self.dropout)
        
        # weighted sum
        results = tf.matmul(attn_w, Value)
        
        # 拼接多头空间
        results = tf.concat(tf.split(results, self.attention_head_size,), axis = -1)
        results = tf.squeeze(results, axis = 0)
        
        # 加上残差网络保留一些原始特征
        if self.use_residual:
            results = results + tf.tensordot(x, self.W_R, axes = (-1,0))
        results = tf.nn.relu(results)
        
        return results
    

class FMLayer(Layer):
    """
        FM Layer
    """
    def __init__(self):
        super(FMLayer, self).__init__()
    
    def build(self, input_shape):
        pass
    
    def call(self, x):
        """
            x : batch, field_dim, embed_dim
        """
        tmp = x
        square_of_sum = tf.square(tf.reduce_sum(tmp, axis = 1))
        sum_of_square = tf.reduce_sum(x * x, axis= 1)
        cross_part = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis = 1, keepdims = True)
        
        return cross_part
    
    
class BitWiseFEN(Layer):
    """
        Bit Wise Net Layer in DIFM
    """
    def __init__(self, input_dim, mlp_dims = [128, 64], dropout = 0.0):
        super(BitWiseFEN, self).__init__()
        
        self.mlp = tf.keras.Sequential()
        for mlp_dim in mlp_dims:
            self.mlp.add(layers.Dense(mlp_dim, input_shape = [input_dim,]))
            self.mlp.add(layers.BatchNormalization())
            self.mlp.add(layers.Activation('relu'))
            self.mlp.add(layers.Dropout(dropout))
            input_dim = mlp_dim     
    
    def build(self, input_shape):
        pass
    
    def call(self, x):
        """
            x : batch, field * embed_dim
        """
        
        return self.mlp(x)


class FENLayer(Layer):
    """
        Dual-FEN Layer in DIFM
    """
    def __init__(self, field_dim, embed_dim, head_num, mlp_dims = [128, 64], dropout=0.0):
        super(FENLayer, self).__init__()
        
        self.bit_wise_net = BitWiseFEN(input_dim = field_dim * embed_dim,
                                       mlp_dims = mlp_dims,
                                       dropout = dropout)
        
        self.vec_wise_net = MultiHeadAttentionInteract(embed_size = embed_dim, 
                                                       head_num = head_num, 
                                                       dropout = dropout)
        
        # Combination Layer
        # 权重最后要转换为长度为field_dim的向量
        self.trans_bit_layer = layers.Dense(field_dim, input_shape = [mlp_dims[-1],])
        self.trans_vec_layer = layers.Dense(field_dim, input_shape = [field_dim * embed_dim,])
        
    def build(self, input_shape):
        pass
    
    def call(self, x):
        """
            x : batch, field_dim, embed_dim
        """
        b, f, e = x.shape
        bit_wise_x = self.bit_wise_net(tf.reshape(x, shape=(b, f*e)))
        vec_wise_x = tf.reshape(self.vec_wise_net(x), shape = (b, f*e))
        
        m_bit = self.trans_bit_layer(bit_wise_x)
        m_vec = self.trans_vec_layer(vec_wise_x)
        
        m_x = m_vec + m_bit
        
        return m_x


class DIFM(Model):
    """
        A Dual Input-aware Factorization Machine for CTR Prediction
    """
    def __init__(self, feature_fields, embed_dim, head_num, mlp_dims = [128, 64], dropout = 0.0):
        super(DIFM, self).__init__()
        field_dim = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        
        # embedding中的weights, First Order
        self.sparse_weight = layers.Embedding(sum(feature_fields), 
                                              1,
                                              input_length = field_dim)
        
        # embedding layer
        self.embedding = layers.Embedding(sum(feature_fields), 
                                          embed_dim, 
                                          input_length = field_dim)
        
        # FEN Layer
        self.FEN = FENLayer(field_dim = field_dim, 
                            embed_dim = embed_dim, 
                            head_num = head_num,
                            mlp_dims = mlp_dims)
        
        # FM layer
        self.FM = FMLayer()
    
    def build(self, input_shape):
        pass
    
    def call(self, x):
        """
        """
        tmp = x + self.offsets
        
        # first order weights
        sparse_weights = self.sparse_weight(tmp)
        # embedding
        embed_x = self.embedding(tmp)
        # m_x
        m_x = self.FEN(embed_x)
        
        # Reweighting
        # w * x
        sparse_weights = tf.squeeze(sparse_weights) * m_x
        sparse_weights = tf.reduce_sum(sparse_weights, axis = 1, keepdims = True)
        # <vxi, vxj>(xi, xj)
        fm_input = embed_x * tf.expand_dims(m_x, axis = 2)
        fm_out = self.FM(fm_input)
        
        # Prediicting
        logit = sparse_weights + fm_out
        
        return tf.sigmoid(logit)
    
        
        
        
        
        
        
        
