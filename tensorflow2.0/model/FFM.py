# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:48:23 2021

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

class FieldFactorizationMachine(Model):
    """
        Field Factorization Machine
    """
    def __init__(self, feature_fields, embed_dim):
        """
        """
        super(FieldFactorizationMachine, self).__init__()
        self.input_dims = sum(feature_fields) + 1
        self.input_lens = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        self.linear = layers.Embedding(self.input_dims,
                                       1, 
                                       input_length = self.input_lens)
        self.embeddings = [
                    layers.Embedding(self.input_dims, 
                                          embed_dim, 
                                          input_length = self.input_lens)
                            for _ in range(self.input_lens)
                        ]
    
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
        # 线性部分
        linear_part = tf.reduce_sum(self.linear(x), axis = 1) + self.bias
        # ffm 内积项
        xs = [self.embeddings[i](x) for i in range(self.input_lens)]
        ix = []
         
        for i in range(self.input_lens - 1):
            for j in range(i+1, self.input_lens):
                ix.append(xs[j][:,i] * xs[i][:,j])
        
        ix = tf.stack(ix, axis = 1)
        ffm_part = tf.reduce_sum(tf.reduce_sum(ix, axis = 1), axis = 1, keepdims = True)
        
        x = linear_part + ffm_part
        x = tf.sigmoid(x)

        return x

