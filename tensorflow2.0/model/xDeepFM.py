# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 16:31:43 2021

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.python.keras.regularizers import l2


class CIN(Model):
    """
        Compressed Interaction Network.
    """
    
    def __init__(self, field_nums, cross_layer_sizes, l2_reg = 1e-5, split_half = True):
        """
            cross_layer_sizes : [H_1, H_2 ,..., H_k], a list of the number of layers
        """
        super(CIN, self).__init__()
        
        self.field_nums = field_nums
        self.cross_layers = cross_layer_sizes
        self.split_half = split_half
        self.l2_reg = l2_reg
    
    def build(self, input_shape):
        
        self.filters = []
        self.bias = []
        self.field_nums = [self.field_nums]
        
        for i, cross_layer_size in enumerate(self.cross_layers):
            
            self.filters.append(
                self.add_weight(name = 'filter_{}'.format(i),
                                shape = [1, self.field_nums[0] * self.field_nums[-1],cross_layer_size],
                                dtype = tf.float32,
                                trainable = True,
                                regularizer = l2(self.l2_reg),
                                )
                )
            
            self.bias.append(
                self.add_weight(name='bias_{}'.format(i), 
                                shape=[cross_layer_size], 
                                dtype=tf.float32,
                                initializer=tf.keras.initializers.Zeros())
                )
            if self.split_half and i != len(self.cross_layers) -1:
                cross_layer_size = (cross_layer_size // 2)
            
            self.field_nums.append(cross_layer_size)
        
        fc_input_dim = sum(self.field_nums[1:])
        
        self.fc = layers.Dense(1, input_shape = [fc_input_dim,])
    
    def call(self, x):
        """
            x : (batch_size, field_dim, embde_dim)
        """
        
        embed_dim = x.shape[-1]
        tmp_results = [x]
        final_res = []
        
        x0 = tf.split(tmp_results[0], embed_dim, 2)
        for i, cross_layer_size in enumerate(self.cross_layers):
            
            x_tmp = tf.split(tmp_results[-1], embed_dim, 2)
            
            res = tf.matmul(x0, x_tmp, transpose_b = True)
            res = tf.reshape(res, shape = [embed_dim, -1, self.field_nums[0] * self.field_nums[i]])
            res = tf.transpose(res, perm = [1,0,2])
        
            res = tf.nn.conv1d(res, filters = self.filters[i], 
                               stride = 1, padding = 'VALID')
            res = tf.nn.bias_add(res, self.bias[i])
            res = tf.nn.relu(res)
            res = tf.transpose(res, perm=[0, 2, 1])
            
            if self.split_half:
                if i != len(self.cross_layers) - 1:
                    next_tmp, h = tf.split(res, 2 * [cross_layer_size // 2], 1)
                else:
                    h = res
                    next_tmp = 0
            else:
                h = res
                next_tmp = res
            
            final_res.append(h)
            tmp_results.append(next_tmp)
        
        results = tf.concat(final_res, axis=1)
        results = tf.reduce_sum(results, -1, keepdims = False)
        
        results = self.fc(results)
        
        return results
            
        
class xDeepFM(Model):
    """
        xDeepFM
    """
    def __init__(self,feature_fields, embed_dim, mlp_dims, 
                 dropout, cross_layer_sizes, split_half = True):
        
        super(xDeepFM, self).__init__()
        
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
        
        #CIN
        self.cin = CIN(field_nums = self.input_lens, 
                       cross_layer_sizes = cross_layer_sizes)
    
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
        
        #交叉层
        mlp_part = self.mlp(tf.reshape(x, (-1, self.embed_out_dim)))
        cin_part = self.cin(x)
        
        x = linear_part + cin_part + mlp_part
        x = tf.sigmoid(x)
        
        return x

       