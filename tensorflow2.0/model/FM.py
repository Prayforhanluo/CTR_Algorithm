# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 15:11:12 2021

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

class FactorizationMachine(Model):
    """
        Factorization Machine
    """
    def __init__(self, feature_fields, embed_dim):
        """
        """
        super(FactorizationMachine, self).__init__()
        self.input_dims = sum(feature_fields) + 1
        self.input_lens = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        self.linear = layers.Embedding(self.input_dims,
                                       1, 
                                       input_length = self.input_lens)
        self.embedding = layers.Embedding(self.input_dims, 
                                          embed_dim, 
                                          input_length = self.input_lens)
    
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
         # 内积项
         x = self.embedding(x)
         square_of_sum = tf.square(tf.reduce_sum(x, axis=1, keepdims = True))
         sum_of_square = tf.reduce_sum(x * x, axis=1, keepdims = True)
         cross_part = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=2, keepdims=False)
         
         x = linear_part + cross_part
         x = tf.sigmoid(x)
         return x
     
        
        
        
        
    