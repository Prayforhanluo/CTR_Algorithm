# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:54:55 2021

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
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



class AutoInt(Model):
    """
        Automatic Feature Interaction via self-Attention
    """
    def __init__(self, feature_fields, embed_dim, head_num,
                 attn_layers, mlp_dims, dropout):
        
        super(AutoInt, self).__init__()
        self.input_dims = sum(feature_fields) + 1
        self.input_lens = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        
        #线性部分
        self.linear = layers.Embedding(self.input_dims, 
                                       1, 
                                       input_length = self.input_lens)
        #embedding
        self.embedding = layers.Embedding(self.input_dims,
                                          embed_dim,
                                          input_length = self.input_lens)
        
        #DNN
        self.embed_out_dim = len(feature_fields) * embed_dim
        input_dim = self.embed_out_dim
        self.mlp = tf.keras.Sequential()
        for mlp_dim in mlp_dims:
            self.mlp.add(layers.Dense(mlp_dim, input_shape = [input_dim,]))
            self.mlp.add(layers.BatchNormalization())
            self.mlp.add(layers.Activation('relu'))
            self.mlp.add(layers.Dropout(dropout))
            input_dim = mlp_dim
        self.mlp.add(layers.Dense(1, input_shape = [input_dim,]))
        
        #MultiHeadAttention layer
        self.attn_output_dim = self.input_lens * embed_dim
        self.attns = tf.keras.Sequential()
        for i in range(attn_layers):
            self.attns.add(MultiHeadAttentionInteract(embed_size = embed_dim, 
                                                      head_num = head_num, 
                                                      dropout = dropout))
        self.attn_fc = layers.Dense(1, input_shape = [self.attn_output_dim,])        
        
    def build(self, input_shape):
        """
            linear 记忆部分的bias项
        """
        self.bias = self.add_weight(shape = (1,),
                                    initializer = 'random_normal',
                                    trainable = True,
                                    name = 'linear_bias',
                                    dtype = tf.float32)

    def call(self, x):
        """
            x : (batch_size, num_fields)
        """
        x = x + self.offsets
        
        #Embedding
        embed_x = self.embedding(x)
        
        #线性部分
        linear_part = tf.reduce_sum(self.linear(x), axis = 1) + self.bias
        
        #Attention部分
        attn_part = self.attn_fc(tf.reshape(self.attns(embed_x),(-1, self.attn_output_dim)))
        
        #DNN部分
        mlp_part = self.mlp(tf.reshape(embed_x, (-1, self.embed_out_dim)))
        
        x = linear_part + attn_part + mlp_part
        x = tf.sigmoid(x)
        
        return x

            

        