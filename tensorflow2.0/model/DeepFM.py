# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 17:44:31 2021

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


class DeepFM(Model):
    """
        Deep Factorization Machine
    """
    def __init__(self, feature_fields, embed_dim, mlp_dims, dropout):
        """
        """
        super(DeepFM, self).__init__()
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
        #线性部分
        linear_part = tf.reduce_sum(self.linear(x), axis = 1) + self.bias
        #embedding
        x = self.embedding(x)
        #二次项交叉
        square_of_sum = tf.square(tf.reduce_sum(x, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(x * x, axis=1, keepdims=True)
        cross_part = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=2, keepdims=False)        
        #DNN结果
        mlp_part = self.mlp(tf.reshape(x, (-1, self.embed_out_dim)))
        
        x = linear_part + cross_part + mlp_part
        x = tf.sigmoid(x)
        
        return x
        
        
        
        