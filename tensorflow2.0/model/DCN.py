# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 16:12:15 2021

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

class DeepCrossNet(Model):
    """
        Deep Cross Network
    """
    def __init__(self, feature_fields, embed_dim, num_layers, mlp_dims, dropout):
        
        super(DeepCrossNet, self).__init__()
        self.feature_fields = feature_fields
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        self.input_dims = sum(feature_fields) + 1
        self.input_lens = len(feature_fields)
        
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
        
        #Cross Net
        self.num_layers = num_layers
        
        #LR
        self.lr = layers.Dense(1, input_shape = [mlp_dims[-1]+self.embed_out_dim,])
    
    def build(self, input_shape):
        """
            自定义显式交叉层的参数
        """
        
        self.cross_w = [
            self.add_weight(name = 'w_{}'.format(i),
                            shape = (self.embed_out_dim, 1),
                            trainable = True,
                            initializer = 'random_normal'
                            )
            for i in range(self.num_layers)
        ]
        
        self.cross_b = [
            self.add_weight(name = 'b_{}'.format(i),
                            shape = (self.embed_out_dim,),
                            initializer = 'random_normal',
                            trainable = True)
            for i in range(self.num_layers)
        ]
    
    def call(self, x):
        x = x + self.offsets
        # embedding
        x = self.embedding(x)
        x = tf.reshape(x, (-1, self.embed_out_dim))
        #DNN out
        mlp_part = self.mlp(x)
        
        #Cross Net out
        x0 = x
        cross = x
        for i in range(self.num_layers):
            xw = tf.tensordot(cross, self.cross_w[i], axes = [1,0])
            cross = x0 * xw + self.cross_b[i] + cross
        
        #Concat output
        out = tf.concat([cross, mlp_part], axis = 1)
        
        #LR predict
        out = self.lr(out)
        out = tf.sigmoid(out)
        
        return out
    
            
        
        