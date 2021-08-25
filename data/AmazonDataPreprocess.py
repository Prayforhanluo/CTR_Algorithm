# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 19:32:16 2021

@author: luoh1
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn import metrics


def AmazonBookPreprocess(dataframe, seq_len = 40):
    """
    亚马逊书数据集的预处理

    Parameters
    ----------
    dataframe : pandas_df
        原始csv的数据框.
    seq_len : int, optional
        用户行为的seq长度. The default is 40.

    Returns
    -------
    

    """
    data = dataframe.copy()
    data['hist_item_list'] = dataframe.apply(lambda x : x['hist_item_list'].split('|'), axis = 1)
    data['hist_cate_list'] = dataframe.apply(lambda x : x['hist_cate_list'].split('|'), axis = 1)
    
    #获取cate的所有种类
    cate_list = list(data['cateID'])
    _ = [cate_list.extend(i) for i in data['hist_cate_list'].values]
    #所有的cate set集合
    cate_set = set(cate_list + ['0']) #用 '0' 作为padding的类别
    
    #截取用户行为的长度,也就是截取hist_cate_list的长度
    cols = ['hist_cate_{}'.format(i) for i in range(seq_len)]

    def trim_cate_list(x):
        if len(x) > seq_len:
            #历史行为大于40, 截取后40个行为
            return pd.Series(x[-seq_len:], index = cols)
        else:
            #历史行为不足40, padding到40个行为
            pad_len = seq_len - len(x)
            x = x + ['0'] * pad_len
            return pd.Series(x, index = cols)
    
    #由于只用了10W条数据, 所以只用了cate的特征, 因为item特征大部分上都是1次出现, 所以暂时不理item特征了
    #同理同时也省略了userID特征
    labels = data['label']
    data = data['hist_cate_list'].apply(trim_cate_list).join(data['cateID'])
    
    #LabelEncode cate 特征
    cate_encoder = LabelEncoder().fit(list(cate_set))
    data = data.apply(cate_encoder.transform).join(labels)
    
    return data