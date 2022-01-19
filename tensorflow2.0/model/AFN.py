# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:31:55 2022

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomNormal, Zeros


class LogTransformLayer(Layer):
    """
        Logarithmic Transformation Layer in AFN        
    """
    
    def __init__(self, field_size, embed_size, hidden_size):
        super(LogTransformLayer, self).__init__()
        
        self.field_size = field_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
    
    def build(self, input_shape):
        
        self.lt_weights = self.add_weight(shape = (self.field_size, self.hidden_size),
                                       initializer = RandomNormal(mean=0,stddev = 0.1),
                                       trainable = True,
                                       name = 'hidden_weight',
                                       dtype = tf.float32)
        
        self.lt_bias = self.add_weight(shape = (1,1,self.hidden_size),
                                    initializer = Zeros(),
                                    trainable = True,
                                    name = 'hidden_bias',
                                    dtype = tf.float32)
    
    def call(self, x):
        """
            x : batch_size, field_size, embed_size
        """
        # log输入要求不能有负数, 所以embedding层的内容都改为正数, 0改为很小的正数
        tmp = tf.abs(x)
        tmp = tf.clip_by_value(tmp, clip_value_min=1e-6, 
                               clip_value_max = tf.convert_to_tensor(np.inf))           
        tmp = tf.transpose(tmp, perm = (0, 2, 1)) # batch_size, embed_size, field_size
        # logarithmic transform        
        tmp = tf.math.log(tmp) # 取对数
        tmp = self.bn1(tmp)
        tmp = tf.matmul(tmp, self.lt_weights) + self.lt_bias # batch_size, embed_size, hidden_size
        tmp = tf.math.exp(tmp)
        tmp = self.bn2(tmp)
        # flatten
        tmp = tf.reshape(tmp, (tmp.shape[0], tmp.shape[1]*tmp.shape[2]))
        
        return tmp
    

class AFN(Model):
    """
        Adaptive Factorization Network
    """
    
    def __init__(self, feature_fields, embed_size, hidden_size, mlp_dims = [128, 64], dropout = 0.1):
        super(AFN, self).__init__()
        self.field_size = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long) 
        self.feature_dims = sum(feature_fields)

        #First Order
        self.linear = layers.Embedding(self.feature_dims, 
                                       1, 
                                       input_length = self.field_size)
        #Embedding
        self.embedding = layers.Embedding(self.feature_dims,
                                          embed_size,
                                          input_length = self.field_size)        
        # AFM Layer
        self.LogTransformLayer = LogTransformLayer(field_size = self.field_size, 
                                                   embed_size = embed_size, 
                                                   hidden_size = hidden_size)
        
        # DNN Layer
        input_dim = embed_size * hidden_size
        self.mlp = tf.keras.Sequential()
        for mlp_dim in mlp_dims:
            self.mlp.add(layers.Dense(mlp_dim, input_shape = [input_dim,]))
            self.mlp.add(layers.BatchNormalization())
            self.mlp.add(layers.Activation('relu'))
            self.mlp.add(layers.Dropout(dropout))
            input_dim = mlp_dim
        self.mlp.add(layers.Dense(1, input_shape = [input_dim,]))       
    
    def build(self, input_shape):
        # First Order Bias
        self.fo_bias = self.add_weight(shape = (1,),
                                    initializer = 'random_normal',
                                    trainable = True,
                                    name = 'linear_bias',
                                    dtype = tf.float32)        
    
    def call(self, x):
        x = x + self.offsets
        # First Order logit
        linear_part = tf.reduce_sum(self.linear(x), axis = 1) + self.fo_bias
        # Embeding
        embed_x = self.embedding(x)
        # Logarithmic transform 
        afn_part = self.LogTransformLayer(embed_x)
        afn_part = self.mlp(afn_part)
        
        res = linear_part + afn_part
        res = tf.sigmoid(res)
        
        return res
    
        
        