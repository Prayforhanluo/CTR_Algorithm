# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:15:16 2021

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

class AttentionalFactorizationMachine(Model):
    """
        Attentional FM
    """
    def __init__(self, feature_fields, embed_dim, attn_size, dropout):
        super(AttentionalFactorizationMachine, self).__init__()
        self.input_dims = sum(feature_fields) + 1
        self.input_lens = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        
        #线性部分定义
        self.linear = layers.Embedding(self.input_dims,
                                       1, 
                                       input_length = self.input_lens)
        #embedding
        self.embedding = layers.Embedding(self.input_dims,
                                          embed_dim,
                                          input_length = self.input_lens)
        #attentions
        self.attention = layers.Dense(attn_size, input_shape = [embed_dim,])
        self.projection = layers.Dense(1, input_shape = [attn_size,])
        self.fc = layers.Dense(1, input_shape = [embed_dim,])
        self.dropout = dropout
        
    
    def build(self, input_shape):
        """
            自定义线性部分的bias偏置项
        """
        self.bias = self.add_weight(shape = (1,),
                                    initializer = 'random_normal',
                                    trainable = True,
                                    name = 'linear_bias',
                                    dtype = tf.float32)
    
    def call(self, x):
        x = x + self.offsets
        linear_part = tf.reduce_sum(self.linear(x), axis = 1) + self.bias
        x = self.embedding(x)
        
        num_fields = x.shape[1]
        row, col = [], []
        for i in range(num_fields - 1):
            for j in range(i+1, num_fields):
                row.append(i)
                col.append(j)
        p = tf.gather(x, axis=1, indices = row)
        q = tf.gather(x, axis=1, indices = col)
        #交叉项
        inner = p * q
        #attention 权重层
        attn_scores = tf.nn.relu(self.attention(inner))
        attn_scores = tf.nn.softmax(self.projection(attn_scores), axis = 1)
        attn_scores = tf.nn.dropout(attn_scores, rate = self.dropout)
        #attention 输出
        attn_outs = tf.reduce_sum(attn_scores * inner, axis = 1)
        attn_outs = tf.nn.dropout(attn_outs, rate = self.dropout)
        inner_attn_part = self.fc(attn_outs)
        
        #最终输出
        x = linear_part + inner_attn_part
        x = tf.sigmoid(x)
        return x
        
        
        