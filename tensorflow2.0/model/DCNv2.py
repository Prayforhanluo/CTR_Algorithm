# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:26:17 2021

@author: luoh1
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.initializers import Zeros

class CrossNetMatrix(Model):
    """
        CrossNet of DCN-v2
    """
    
    def __init__(self, layer_num = 2):
        super(CrossNetMatrix, self).__init__()
        self.layer_num = layer_num
    
    
    def build(self, input_shape):
        
        in_features = int(input_shape[-1])
        # Cross中的W参数 (layer_num,  [W])
        self.cross_weights = [self.add_weight(name = 'Cross_weight_{}'.format(str(i)),
                                        shape = (in_features, in_features),
                                        initializer = 'random_normal',
                                        trainable = True)
                        for i in range(self.layer_num)]

        # Cross中的b参数 (layer_num,  [b])
        self.cross_bias = [self.add_weight(name = 'Cross_bias_{}'.format(str(i)),
                                        shape = (in_features, 1),
                                        initializer = Zeros(),
                                        trainable = True)
                        for i in range(self.layer_num)]
        
    
    def call(self, x):
        """
            x : batch_size  *  in_features
        """
        x0 = tf.expand_dims(x, axis=2)
        xl = x0
        for i in range(self.layer_num):
            tmp = tf.matmul(self.cross_weights[i], xl) + self.cross_bias[i]
            xl = x0 * tmp + xl
        
        xl = tf.squeeze(xl, axis=2)
        
        return xl
    
    
class CrossNetMix(Model):
    """
        CrossNet of DCN-V2 with Mixture of Low-rank Experts
        公式如下：
            G_i(xl) = Linear(xl)
            E_i(xl) = x0·(Ul*g(Cl*g(Vl*xl)) + bl)
            g() = tanh activate func
    """
    
    def __init__(self, low_rank = 16, expert_num = 4, layer_num = 2):
        super(CrossNetMix, self).__init__()
        
        self.layer_num = layer_num
        self.expert_num = expert_num
        self.low_rank = low_rank
        
    
    def build(self, input_shape):
        
        in_features = int(input_shape[-1])
        
        # Cross中的U参数(layer_num, expert_num, in_features, low_rank)
        self.U_params = [self.add_weight(name = 'U_params_{}'.format(str(i)),
                                         shape = (self.expert_num, in_features, self.low_rank),
                                         initializer = 'random_normal',
                                         trainable = True)
                         for i in range(self.layer_num)]
        
        # Cross中的V参数(layer_num, expert_num, low_rank, in_features)
        self.V_params = [self.add_weight(name = 'V_params_{}'.format(str(i)),
                                         shape = (self.expert_num, self.low_rank, in_features),
                                         initializer = 'random_normal',
                                         trainable = True)
                         for i in range(self.layer_num)]
        
        # Cross中的C参数(layer_num, expert_num, low_rank, low_rank)
        self.C_params = [self.add_weight(name = 'C_params_{}'.format(str(i)),
                                         shape = (self.expert_num, self.low_rank, self.low_rank),
                                         initializer = 'random_normal',
                                         trainable = True)
                         for i in range(self.layer_num)]

        # Cross中的b参数 (layer_num,  [b])
        self.cross_bias = [self.add_weight(name = 'Cross_bias_{}'.format(str(i)),
                                        shape = (in_features, 1),
                                        initializer = Zeros(),
                                        trainable = True)
                        for i in range(self.layer_num)]

        # MOE 中的门控gate
        self.gates = [tf.keras.layers.Dense(1, use_bias=False) 
                      for i in range(self.expert_num)]

    def call(self, x):
        """
            x : batch_size  *  in_features
        """
        x0 = tf.expand_dims(x, axis=2)
        xl = x0
        for i in range(self.layer_num):
            expert_out = []
            gate_score = []
            for expert in range(self.expert_num):
                # gate score : G(xl)
                gate_score.append(self.gates[expert](tf.squeeze(xl, axis=2)))                
                
                # cross part
                # g(Vl·xl))
                tmp = tf.nn.tanh(tf.matmul(self.V_params[i][expert], xl))
                # g(Cl·g(Vl·xl))
                tmp = tf.nn.tanh(tf.matmul(self.C_params[i][expert], tmp))
                # Ul·g(Cl·g(Vl·xl)) + bl
                tmp =  tf.matmul(self.U_params[i][expert], tmp) + self.cross_bias[i]
                # E_i(xl) = x0·(Ul·g(Cl·g(Vl·xl)) + bl)
                tmp = x0 * tmp
                expert_out.append(tf.squeeze(tmp, axis = 2))
            
            # mixture of low-rank experts
            expert_out = tf.stack(expert_out, 2)
            gate_score = tf.stack(gate_score, 1)
            MOE_out = tf.matmul(expert_out, tf.nn.softmax(gate_score, 1))
            xl = MOE_out + xl
        
        xl = tf.squeeze(xl, axis=2)
        
        return xl


class DeepCrossNetv2(Model):
    """
        Deep Cross Network V2
    """
    
    def __init__(self, feature_fields, embed_dim, layer_num, mlp_dims, dropout=0.1,
                 cross_method = 'Mix', model_method = 'parallel'):
        super(DeepCrossNetv2, self).__init__()
        self.feature_fields = feature_fields
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        self.input_dims = sum(feature_fields)
        self.input_lens = len(feature_fields) 
        self.cross_method = cross_method
        self.model_method = model_method
    
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
        if cross_method == 'Mix':
            self.CrossNet = CrossNetMix(layer_num=layer_num)
        elif cross_method == 'Matrix':
            self.CrossNet = CrossNetMatrix(layer_num=layer_num)
        else:
            raise NotImplementedError
        
        #Predict
        if model_method == 'parallel':
            self.fc = layers.Dense(1, input_shape = [mlp_dims[-1]+self.embed_out_dim,])
        elif model_method == 'stack':
            self.fc = layers.Dense(1, input_shape = [mlp_dims[-1],])
        else:
            raise NotImplementedError
    
    def call(self, x):
        x = x + self.offsets
        x = self.embedding(x)
        x = tf.reshape(x, (-1, self.embed_out_dim))
        if self.model_method == 'parallel':
            mlp_part = self.mlp(x)
            cross = self.CrossNet(x)
            out = tf.concat([cross, mlp_part], axis = 1)
        elif self.model_method == 'stack':
            cross = self.CrossNet(x)
            out = self.mlp(cross)
        else:
            raise NotImplementedError
        
        out = self.fc(out)
        out = tf.sigmoid(out)
        
        return out        