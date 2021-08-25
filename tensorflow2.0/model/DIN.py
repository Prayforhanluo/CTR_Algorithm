# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 20:15:45 2021

@author: luoh1
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model


class Dice(layers.Layer):
    """
        Dice Activation Layer
    """
    def __init__(self):
        super(Dice, self).__init__()
        self.epsilon = 1e-9
    
    def build(self, input_shape):
        self.alpha = self.add_weight(name = 'dice_alpha',
                                     shape = (),
                                     initializer = keras.initializers.Zeros(),
                                     trainable = True)
    
    def call(self, x):
        
        top = x - tf.reduce_mean(x, axis=0)
        bottom = tf.sqrt(tf.math.reduce_variance(x, axis=0) + self.epsilon)
        norm_x = top / bottom
        p = tf.sigmoid(norm_x)
        x = self.alpha * x * (1-p) + x * p
        return x
        

class ActivationUnit(Model):
    """
        Activation Unit
    """
    def __init__(self, embed_dim, dropout = 0.2, fc_dims = [32, 16]):
        super(ActivationUnit, self).__init__()
        
        self.fc_layers = keras.Sequential()
        input_dim = embed_dim*4
        for fc_dim in fc_dims:
            self.fc_layers.add(layers.Dense(fc_dim, input_shape = [input_dim,]))
            self.fc_layers.add(Dice())
            self.fc_layers.add(layers.Dropout(dropout))
            self.input_dim = fc_dim
        self.fc_layers.add(layers.Dense(1, input_shape = [input_dim,]))
    
    def call(self, query, user_behavior):
        """
            query : 单独的ad的embedding mat -> batch * 1 * embed 
            user_behavior : 行为特征矩阵 -> batch * seq_len * embed            
        """
        
        # repeat ads
        seq_len = user_behavior.shape[1]
        queries = tf.concat([query]*seq_len, axis = 1)
        attn_input = tf.concat([queries,
                                user_behavior,
                                queries - user_behavior,
                                queries * user_behavior], axis = -1)
        out = self.fc_layers(attn_input)
        return out


class AttentionPoolingLayer(Model):
    """
        Attention Pooling Layer
    """
    def __init__(self, embed_dim, dropout):
        super(AttentionPoolingLayer, self).__init__()
        
        self.active_unit = ActivationUnit(embed_dim=embed_dim,
                                          dropout=dropout)
    
    def call(self, query, user_behavior, mask):
        """
            query_ad : 单独的ad的embedding mat -> batch * 1 * embed
            user_behavior : 行为特征矩阵 -> batch * seq_len * embed
            mask : 被padding为0的行为置为false -> batch * seq_len * 1
        """
        
        # attn weights
        attn_weights = self.active_unit(query, user_behavior)
        # mul weights and sum pooling
        output = user_behavior * attn_weights * mask
        output = tf.reduce_sum(output, axis = 1) # batch * embed_dim
        
        return output
    
    
class DeepInterestNet(Model):
    """
        Deep Interest Net
    """
    def __init__(self, feature_dim, embed_dim, mlp_dims, dropout):
        super(DeepInterestNet, self).__init__()
        self.embedding = layers.Embedding(feature_dim+1, embed_dim)
        self.AttentionActivate = AttentionPoolingLayer(embed_dim, dropout)
          
        self.fc_layers = keras.Sequential()
        input_dim = embed_dim*2
        for fc_dim in mlp_dims:
            self.fc_layers.add(layers.Dense(fc_dim, input_shape = [input_dim,]))
            self.fc_layers.add(layers.Activation('relu'))
            self.fc_layers.add(layers.Dropout(dropout))
            self.input_dim = fc_dim
        self.fc_layers.add(layers.Dense(1, input_shape = [input_dim,]))  
    
    def call(self, x):
        """
            x : (behaviors * 40, ads * 1) -> batch * (behaviors + ads)
        """
        # define mask
        behavior_x = x[:,:-1]
        mask = tf.cast(behavior_x > 0, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        
        ads_x = x[:,-1]
        
        # embedding
        query_ad = tf.expand_dims(self.embedding(ads_x), axis=1)
        user_behavior = self.embedding(behavior_x)
        user_behavior = user_behavior * mask
        
        # attn pooling
        user_interest = self.AttentionActivate(query_ad, user_behavior, mask)
        concat_input = tf.concat([user_interest, 
                                  tf.squeeze(query_ad, axis=1)],
                                 axis = 1)
        # MLPs prediction
        out = self.fc_layers(concat_input)
        out = tf.sigmoid(out)
        
        return out
        
        
        