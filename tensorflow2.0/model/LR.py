# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:45:39 2021

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


class LogisticRegression(Model):
    """
        Logistic Regression
    """
    def __init__(self, feature_fields):
        super(LogisticRegression, self).__init__()
        self.input_dims = sum(feature_fields) + 1
        self.input_lens = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        
        self.linear = layers.Embeding(self.input_dims, 1, input_length = self.input_lens)
        
    def build(self, input_shape):
        """
            自定义bias偏置项
        """
        self.bias = self.add_weight(shape = (1,),
                                    initializer = 'random_normal',
                                    trainable = True,
                                    name = 'linear_bias',
                                    dtype = tf.float32)
    
    def call(self, x):
        x = x + self.offsets
        x = tf.reduce_sum(self.linear(x), axis = 1) + self.bias
        x = tf.sigmoid(x)
        return x
    
