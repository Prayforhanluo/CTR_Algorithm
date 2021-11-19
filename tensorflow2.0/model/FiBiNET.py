# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:46:01 2021

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

class SELayer(Layer):
    """
        Squenzee and Excitation Layer in FiBiNET
    """
    
    def __init__(self, field_size, reduction_ratio, pooling = 'mean'):
        super(SELayer, self).__init__()
        self.field_size = field_size
        self.reduction_ratio = reduction_ratio
        self.pooling = pooling
        
        self.reduction_size = max(1, field_size // reduction_ratio)
        
        self.excitation = tf.keras.Sequential()
        
        self.excitation.add(layers.Dense(self.reduction_size, 
                                         input_shape =(field_size,),
                                         use_bias = False))
        self.excitation.add(layers.Activation('relu'))
        self.excitation.add(layers.Dense(field_size,
                                         input_shape = (self.reduction_size,),
                                         use_bias = False))
        self.excitation.add(layers.Activation('relu'))
    
    def call(self, x):
        """
            x : batch * field_size * embed_dim
        """
        
        # squeeze
        if self.pooling == 'mean':
            z = tf.reduce_mean(x, axis = -1)
        elif self.pooling == 'max':
            z = tf.reduce_max(x, axis = -1)
        else:
            raise NotImplementedError
        # excitation
        A = self.excitation(z)
        
        # reweight embedding
        V = tf.multiply(x, tf.expand_dims(A, axis=2))
        
        return V
    
class BilinearInteraction(Layer):
    """
        BilinearInteraction Layer in FiBiNET
    """
    
    def __init__(self, field_size, embed_dim, bilinear_type = 'interaction'):
        super(BilinearInteraction, self).__init__()
        
        self.field_size = field_size
        self.embed_dim = embed_dim
        self.bilinear_type = bilinear_type
    
    def build(self, input_shape):
        
        if self.bilinear_type == 'all':
            self.W = self.add_weight(name = 'bilinear_W',
                                     shape = (self.embed_dim, self.embed_dim),
                                     initializer = 'random_normal',
                                     trainable = True)
        elif self.bilinear_type == 'each':
            self.W_List = [self.add_weight(name = 'bilinear_W_{}'.format(i),
                                           shape = (self.embed_dim, self.embed_dim),
                                           initializer = 'random_normal',
                                           trainable = True)
                           for i in range(self.field_size)]
        elif self.bilinear_type == 'interaction':
            self.W_List = [self.add_weight(name = 'bilinear_W_{}'.format(i),
                                           shape = (self.embed_dim, self.embed_dim),
                                           initializer = 'random_normal',
                                           trainable = True)
                           for i in range(self.field_size)
                               for j in range(i+1, self.field_size)]
        else:
            raise NotImplementedError
    
    def call(self, x):
        """
            x : batch * field_size * embed_dim
        """
        x = tf.split(x, self.field_size, axis=1) # 将每个field拿出来
        p = []
        if self.bilinear_type == 'all':
            for i in range(self.field_size):
                v_i = x[i]
                for j in range(i+1, self.field_size):
                    v_j = x[j]
                    tmp = tf.tensordot(v_i, self.W, axes=(-1,0))
                    tmp = tf.multiply(tmp, v_j)
                    p.append(tmp)
        elif self.bilinear_type == 'each':
            for i in range(self.field_size):
                v_i = x[i]
                for j in range(i+1, self.field_size):
                    v_j = x[j]
                    tmp = tf.tensordot(v_i, self.W_List[i], axes=(-1,0))
                    tmp = tf.multiply(tmp, v_j)
                    p.append(tmp)                    
        elif self.bilinear_type == 'interaction':
            num = 0 # 取 Vi, Vj 对应的W
            for i in range(self.field_size):
                v_i = x[i]
                for j in range(i+1, self.field_size):
                    v_j = x[j]
                    tmp = tf.tensordot(v_i, self.W_List[num], axes=(-1,0))
                    tmp = tf.multiply(tmp, v_j)
                    p.append(tmp)
                    num += 1
        else:
            raise NotImplementedError
        
        p = tf.concat(p, axis = 1)
        
        return p
    

class FiBiNET(Model):
    """
        Feature Importance and Bilinear feature Interaction Net
    """
    
    def __init__(self, feature_fields, embed_dim, reduction_ratio,
                 pooling = 'mean', mlp_dims = (64, 32), dropout=0.):
        super(FiBiNET, self).__init__()
        
        self.field_size = len(feature_fields)
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        
        # Embedding Layer
        self.embedding = layers.Embedding(sum(feature_fields) + 1,
                                          embed_dim,
                                          input_length = self.field_size)
        
        # SE Layer
        self.SELayer = SELayer(self.field_size, reduction_ratio)
        
        # Bilinear Layer
        self.Bilinear = BilinearInteraction(self.field_size, embed_dim)
        
        # Final DNN
        self.embed_out_dim = self.field_size * (self.field_size - 1) * embed_dim
        input_dim = self.embed_out_dim
        self.mlp = tf.keras.Sequential()
        for mlp_dim in mlp_dims:
            self.mlp.add(layers.Dense(mlp_dim, input_shape = [input_dim,]))
            self.mlp.add(layers.BatchNormalization())
            self.mlp.add(layers.Activation('relu'))
            self.mlp.add(layers.Dropout(dropout))
            input_dim = mlp_dim
        self.mlp.add(layers.Dense(1, input_shape = [input_dim,]))        
        
    
    def call(self, x):
        """
            x : batch * field_size
        """
        
        x = x + self.offsets
        # embedding_x
        embed_x = self.embedding(x)
    
        # SENet-like embedding
        SE_embed_x = self.SELayer(embed_x)
        batch = tf.shape(embed_x)[0]
        
        # Bilinear interaction
        p = tf.reshape(self.Bilinear(embed_x), (batch, -1))
        se_p = tf.reshape(self.Bilinear(SE_embed_x), (batch, -1))
        
        # final DNN
        concat_p = tf.concat([p, se_p], axis=1)
        res = tf.sigmoid(self.mlp(concat_p))
        return res
        
        
            