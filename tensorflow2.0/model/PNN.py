# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:13:16 2021

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

class InnerProductNetwork(Model):
    """
        Inner Product
    """
    
    def call(self, x):
        """
        x : (batch_size, num_fields, embed_dim)
        """
        num_fields = x.shape[1]
        row, col = [],[]
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i)
                col.append(j)
        p = tf.gather(x, axis = 1, indices = row)
        q = tf.gather(x, axis = 1, indices = col)
        return tf.reduce_sum(p * q, axis = 2)


class OuterProductNetwork(Model):
    """
        Outer Product
    """
    
    def __init__(self, num_fields, embed_dim, kernel_type = 'mat'):
        super(OuterProductNetwork, self).__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_shape == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('Unknown kernel type : ' + kernel_type)
        
        self.kernel_type = kernel_type
        self.kernel_shape = kernel_shape
    
    def build(self, input_shape):
        
        self.kernel = self.add_weight(shape = self.kernel_shape,
                                      initializer = 'random_normal',
                                      trainable = True,
                                      name = 'outer_weight',
                                      dtype = tf.float32)
    def call(self, x):
        """
        x : (batch_size, num_fields, embed_dim)
        """
        num_fields = x.shape[1]
        row, col = [], []
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i)
                col.append(j)
        p = tf.gather(x, axis = 1, indices = row)
        q = tf.gather(x, axis = 1, indices = col)
        if self.kernel_type == 'mat':
            tmp = tf.expand_dims(p, axis = 1) * self.kernel
            tmp = tf.reduce_sum(tmp, axis = -1)
            tmp = tf.transpose(tmp, perm = (0,2,1))
            kp = tf.reduce_sum(tmp * q, axis = -1)
        else:
            tmp = p * q * tf.expand_dims(self.kernel, axis=0)
            kp = tf.reduce_sum(tmp, axis = -1)
        
        return kp
    

class PNN(Model):
    """
        Product-Based Neural Network
    """
    def __init__(self, feature_fields, embed_dim, mlp_dims, dropout, method = 'inner'):
        
        super(PNN, self).__init__()
        self.feature_fields = feature_fields
        self.input_dims = sum(feature_fields) + 1
        self.input_lens = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        if method not in ['inner', 'outer']:
            raise ValueError('Unknown kernel type : %s' % method)
        else:
            self.method = method
        
        # Embedding
        self.embedding = layers.Embedding(self.input_dims,
                                          embed_dim,
                                          input_length = self.input_lens)
        self.embed_out_dim = self.input_lens * embed_dim
        
        # Product layer
        if self.method == 'inner':
            self.pn = InnerProductNetwork()
        else:
            self.pn = OuterProductNetwork(self.input_lens, embed_dim)
        
        # DNN layer
        input_dim = self.embed_out_dim + \
                    (self.input_lens * (self.input_lens - 1)) // 2
                    
        self.mlp = tf.keras.Sequential()
        for mlp_dim in mlp_dims:
            self.mlp.add(layers.Dense(mlp_dim, input_shape = [input_dim,]))
            self.mlp.add(layers.BatchNormalization())
            self.mlp.add(layers.Activation('relu'))
            self.mlp.add(layers.Dropout(dropout))
            input_dim = mlp_dim
        self.mlp.add(layers.Dense(1, input_shape = [input_dim,]))        
        
        
    def call(self, x):
        x = x + self.offsets
        # embedding
        x = self.embedding(x)
        # product connection
        product_part = self.pn(x)
        x = tf.concat([tf.reshape(x, (-1, self.embed_out_dim)), product_part], axis = 1)
        
        x = self.mlp(x)
        x = tf.sigmoid(x)
        return x
        
        
        
        
        
        
        