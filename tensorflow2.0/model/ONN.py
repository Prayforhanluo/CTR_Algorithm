# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:53:42 2022

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model


class ONN(Model):
    """
        Operation-aware Neural Networks
        FFM & PNN = ONN
        刚开始读ONN论文，感觉就像灌水模型，FFM 跟 PNN 拼接成了 ONN
        这个模型的参数量远超其他
        在embedding的时候，每个field的embedding在做不同操作的时候，都会不一样
        个人觉得简单来说就是粗暴的拓宽了假设空间
        这个模型给我的最大的启示是：
        O(n2) > O(n)
        简单粗暴的增加模型复杂度的确有好处
        缺点就是慢！！！
    """
    
    def __init__(self, feature_fields, embed_dim, mlp_dims, dropout):
        super(ONN, self).__init__()
        self.feature_files = feature_fields
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        self.num_fields = len(feature_fields)
        
        # embeddings
        self.embeddings = [
                    layers.Embedding(sum(feature_fields),
                                     embed_dim,
                                     input_length=self.num_fields)
                    for _ in range(self.num_fields)
                    ]

        # DNN part
        input_dim = embed_dim * self.num_fields + \
                    (self.num_fields * (self.num_fields - 1)) // 2
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
        """
        x = x + self.offsets     
        # embedding matrix
        field_aware_embeds = [embedding(x) for embedding in self.embeddings]
        # raw embedding
        raw_embed = field_aware_embeds[-1] #最后一项embedding 用作raw embedding.
        B, T, E = raw_embed.shape
        raw_embed = tf.reshape(raw_embed, (B, -1, ))
        # field-aware inner product
        interaction = []
        for i in range(self.num_fields - 1):
            for j in range(i+1, self.num_fields):
                tmp_i_j = field_aware_embeds[j-1][:,i,:]
                tmp_j_i = field_aware_embeds[i][:,j,:]
                tmp_dot = tf.reduce_sum(tmp_i_j*tmp_j_i, axis = 1, keepdims=True)
                interaction.append(tmp_dot)
        ffm_out = tf.concat(interaction, axis = 1)
        # DNN
        dnn_input = tf.concat([raw_embed, ffm_out], axis = 1)
        dnn_part = self.mlp(dnn_input)
        out = tf.sigmoid(dnn_part)
        
        return out
        
        
        
        
        