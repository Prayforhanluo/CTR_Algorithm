```python
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from tqdm import tqdm
```


```python
tf.get_logger().setLevel('ERROR')
```

## Loading Data


```python
data = pd.read_csv('../data/data.csv')
```


```python
data_X = data.iloc[:,2:]
data_y = data.click.values
```


```python
data_X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour</th>
      <th>C1</th>
      <th>banner_pos</th>
      <th>site_id</th>
      <th>site_domain</th>
      <th>site_category</th>
      <th>app_id</th>
      <th>app_domain</th>
      <th>app_category</th>
      <th>device_id</th>
      <th>...</th>
      <th>device_type</th>
      <th>device_conn_type</th>
      <th>C14</th>
      <th>C15</th>
      <th>C16</th>
      <th>C17</th>
      <th>C18</th>
      <th>C19</th>
      <th>C20</th>
      <th>C21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>1fbe01fe</td>
      <td>f3845767</td>
      <td>28905ebd</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>07d7df22</td>
      <td>a99f214a</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>15705</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>4dd0a958</td>
      <td>79cf0c8d</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>07d7df22</td>
      <td>a99f214a</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>20352</td>
      <td>320</td>
      <td>50</td>
      <td>2333</td>
      <td>0</td>
      <td>39</td>
      <td>-1</td>
      <td>157</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>543a539e</td>
      <td>c7ca3108</td>
      <td>3e814130</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>07d7df22</td>
      <td>a99f214a</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>20352</td>
      <td>320</td>
      <td>50</td>
      <td>2333</td>
      <td>0</td>
      <td>39</td>
      <td>-1</td>
      <td>157</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>8cbacf0b</td>
      <td>a434fa42</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>07d7df22</td>
      <td>a99f214a</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>19772</td>
      <td>320</td>
      <td>50</td>
      <td>2227</td>
      <td>0</td>
      <td>687</td>
      <td>100075</td>
      <td>48</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14102100</td>
      <td>1005</td>
      <td>0</td>
      <td>f282ab5a</td>
      <td>61eb5bc4</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>07d7df22</td>
      <td>a99f214a</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>18993</td>
      <td>320</td>
      <td>50</td>
      <td>2161</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>157</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>14102101</td>
      <td>1005</td>
      <td>0</td>
      <td>1fbe01fe</td>
      <td>f3845767</td>
      <td>28905ebd</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>07d7df22</td>
      <td>a99f214a</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>15705</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>100084</td>
      <td>79</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>14102101</td>
      <td>1005</td>
      <td>1</td>
      <td>d9750ee7</td>
      <td>98572c79</td>
      <td>f028772b</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>07d7df22</td>
      <td>a99f214a</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>17614</td>
      <td>320</td>
      <td>50</td>
      <td>1993</td>
      <td>2</td>
      <td>1063</td>
      <td>-1</td>
      <td>33</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>14102101</td>
      <td>1005</td>
      <td>0</td>
      <td>85f751fd</td>
      <td>c4e18dd6</td>
      <td>50e219e0</td>
      <td>febd1138</td>
      <td>82e27996</td>
      <td>0f2161f8</td>
      <td>a99f214a</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>21611</td>
      <td>320</td>
      <td>50</td>
      <td>2480</td>
      <td>3</td>
      <td>297</td>
      <td>100111</td>
      <td>61</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>14102101</td>
      <td>1005</td>
      <td>0</td>
      <td>1fbe01fe</td>
      <td>f3845767</td>
      <td>28905ebd</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>07d7df22</td>
      <td>a99f214a</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>15699</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>100084</td>
      <td>79</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>14102101</td>
      <td>1005</td>
      <td>0</td>
      <td>1fbe01fe</td>
      <td>f3845767</td>
      <td>28905ebd</td>
      <td>ecad2386</td>
      <td>7801e8d9</td>
      <td>07d7df22</td>
      <td>a99f214a</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>15706</td>
      <td>320</td>
      <td>50</td>
      <td>1722</td>
      <td>0</td>
      <td>35</td>
      <td>-1</td>
      <td>79</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 22 columns</p>
</div>



    可以看到测试的数据全都是类别特征, 其实实际的业务场景中几乎也都是类别型的特征
    这里我们给特征进行Label Encode


```python
data_X = data_X.apply(LabelEncoder().fit_transform)
```


```python
data_X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour</th>
      <th>C1</th>
      <th>banner_pos</th>
      <th>site_id</th>
      <th>site_domain</th>
      <th>site_category</th>
      <th>app_id</th>
      <th>app_domain</th>
      <th>app_category</th>
      <th>device_id</th>
      <th>...</th>
      <th>device_type</th>
      <th>device_conn_type</th>
      <th>C14</th>
      <th>C15</th>
      <th>C16</th>
      <th>C17</th>
      <th>C18</th>
      <th>C19</th>
      <th>C20</th>
      <th>C21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>110</td>
      <td>823</td>
      <td>1</td>
      <td>712</td>
      <td>28</td>
      <td>0</td>
      <td>5703</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>128</td>
      <td>3</td>
      <td>2</td>
      <td>36</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>303</td>
      <td>403</td>
      <td>16</td>
      <td>712</td>
      <td>28</td>
      <td>0</td>
      <td>5703</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>303</td>
      <td>3</td>
      <td>2</td>
      <td>103</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>334</td>
      <td>668</td>
      <td>3</td>
      <td>712</td>
      <td>28</td>
      <td>0</td>
      <td>5703</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>303</td>
      <td>3</td>
      <td>2</td>
      <td>103</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>543</td>
      <td>563</td>
      <td>16</td>
      <td>712</td>
      <td>28</td>
      <td>0</td>
      <td>5703</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>234</td>
      <td>3</td>
      <td>2</td>
      <td>76</td>
      <td>0</td>
      <td>26</td>
      <td>53</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>924</td>
      <td>316</td>
      <td>16</td>
      <td>712</td>
      <td>28</td>
      <td>0</td>
      <td>5703</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>210</td>
      <td>3</td>
      <td>2</td>
      <td>71</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>31</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>110</td>
      <td>823</td>
      <td>1</td>
      <td>712</td>
      <td>28</td>
      <td>0</td>
      <td>5703</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>128</td>
      <td>3</td>
      <td>2</td>
      <td>36</td>
      <td>0</td>
      <td>1</td>
      <td>59</td>
      <td>18</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>825</td>
      <td>510</td>
      <td>16</td>
      <td>712</td>
      <td>28</td>
      <td>0</td>
      <td>5703</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>173</td>
      <td>3</td>
      <td>2</td>
      <td>60</td>
      <td>2</td>
      <td>30</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>519</td>
      <td>658</td>
      <td>5</td>
      <td>767</td>
      <td>31</td>
      <td>2</td>
      <td>5703</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>407</td>
      <td>3</td>
      <td>2</td>
      <td>130</td>
      <td>3</td>
      <td>13</td>
      <td>77</td>
      <td>13</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>110</td>
      <td>823</td>
      <td>1</td>
      <td>712</td>
      <td>28</td>
      <td>0</td>
      <td>5703</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>122</td>
      <td>3</td>
      <td>2</td>
      <td>36</td>
      <td>0</td>
      <td>1</td>
      <td>59</td>
      <td>18</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>110</td>
      <td>823</td>
      <td>1</td>
      <td>712</td>
      <td>28</td>
      <td>0</td>
      <td>5703</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>129</td>
      <td>3</td>
      <td>2</td>
      <td>36</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 22 columns</p>
</div>



    每一个特征都独立进行了label 编码， 这种好处是可以直接进行embedding
    当我们embedding共享权值的时候， 可以给每列特征的label加入之前特征的类别总和，来达到所有特征的label
    这也是所有模型代码中 offset 的作用
    
    e.g. field_dims = [2, 4, 2], offsets = [0, 2, 6]

    所以，实际look up table中
    0 - 1行 对应 特征 X0, 即 field_dims[0]
    2 - 5行 对应 特征 X1, 即 field_dims[1]
    6 - 7行 对应 特征 X2, 即 field_dims[2]
    但实际特征取值 forward(self, x) 的 x大小 只在自身词表内取值
    比如：X1取值1，对应Embedding内行数就是 offsets[X1] + X1 = 2 + 1 = 3



```python
fields = data_X.max().values + 1 # 模型输入的feature_fields
```


```python
fields
```




    array([    2,     6,     6,   987,   872,    18,   769,    62,    19,
            8544, 47309,  2606,     4,     4,   448,     5,     6,   141,
               4,    38,   144,    33], dtype=int64)




```python
tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size = 0.2, random_state=42, stratify=data_y)
train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size = 0.25, random_state=42, stratify=tmp_y)
```

## 训练过程
   
    数据是avazu数据的随机10万条
    优化器统一Adam， lr = 0.001
    epoch 为 1, batch_size = 32
    主要的目的是跑通所有的模型
    epoch多几次, 调调参数对稍微复杂的网络有好处
    
    
    tips : 类别特征embedding等价于一层没有bias项的全连接，所以模型中几乎都用embedding来模拟LR线性过程

#### LR


```python
from model import LR
```


```python
model = LR.LogisticRegression(feature_fields = fields)
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    Train on 60000 samples, validate on 20000 samples
    60000/60000 [==============================] - 5s 86us/sample - loss: 0.4232 - auc: 0.7029 - val_loss: 0.4135 - val_auc: 0.7317
    




    <tensorflow.python.keras.callbacks.History at 0x243c491bdd8>



#### FM


```python
from model import FM
```


```python
model = FM.FactorizationMachine(feature_fields = fields, embed_dim = 8)
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    Train on 60000 samples, validate on 20000 samples
    60000/60000 [==============================] - 14s 229us/sample - loss: 0.4195 - auc_1: 0.7145 - val_loss: 0.4058 - val_auc_1: 0.7435
    




    <tensorflow.python.keras.callbacks.History at 0x243cb3ecc18>



#### FFM


```python
from model import FFM
```


```python
model = FFM.FieldFactorizationMachine(feature_fields = fields, embed_dim = 8)
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    Train on 60000 samples, validate on 20000 samples
    60000/60000 [==============================] - 216s 4ms/sample - loss: 0.4079 - auc_2: 0.7364 - val_loss: 0.4018 - val_auc_2: 0.7529
    




    <tensorflow.python.keras.callbacks.History at 0x243d1f00908>



#### AFM


```python
from model import AFM
```


```python
model = AFM.AttentionalFactorizationMachine(feature_fields = fields, embed_dim = 8, attn_size = 8, dropout = 0.2)
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    Train on 60000 samples, validate on 20000 samples
    60000/60000 [==============================] - 20s 325us/sample - loss: 0.4267 - auc_3: 0.6965 - val_loss: 0.4119 - val_auc_3: 0.7317
    




    <tensorflow.python.keras.callbacks.History at 0x243ec9d36d8>



#### DeepFM


```python
from model import DeepFM
```


```python
model = DeepFM.DeepFM(feature_fields = fields, embed_dim = 8, mlp_dims = [32,16], dropout=0.2)
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    Train on 60000 samples, validate on 20000 samples
    60000/60000 [==============================] - 17s 288us/sample - loss: 0.4250 - auc_4: 0.7027 - val_loss: 0.4068 - val_auc_4: 0.7410
    




    <tensorflow.python.keras.callbacks.History at 0x243cc90a438>



#### xDeepFM


```python
from model import xDeepFM
```


```python
model = xDeepFM.xDeepFM(feature_fields = fields, embed_dim = 8, mlp_dims = (32, 16), 
                        dropout = 0.3, cross_layer_sizes = (16, 16))
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    Train on 60000 samples, validate on 20000 samples
    60000/60000 [==============================] - 20s 340us/sample - loss: 0.4291 - auc_5: 0.6972 - val_loss: 0.4099 - val_auc_5: 0.7386
    




    <tensorflow.python.keras.callbacks.History at 0x243f5982128>



#### PNN


```python
from model import PNN
```


```python
model = PNN.PNN(feature_fields=fields, embed_dim=8, mlp_dims=[32,16], dropout=0.2, method='inner')
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    Train on 60000 samples, validate on 20000 samples
    60000/60000 [==============================] - 15s 246us/sample - loss: 0.4352 - auc_6: 0.6846 - val_loss: 0.4120 - val_auc_6: 0.7339
    




    <tensorflow.python.keras.callbacks.History at 0x243fb894080>



#### DCN


```python
from model import DCN
```


```python
model = DCN.DeepCrossNet(feature_fields=fields, embed_dim=8, num_layers=3, mlp_dims=[32, 16], dropout=0.2)
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    Train on 60000 samples, validate on 20000 samples
    60000/60000 [==============================] - 14s 241us/sample - loss: 0.4152 - auc_7: 0.7203 - val_loss: 0.4049 - val_auc_7: 0.7458
    




    <tensorflow.python.keras.callbacks.History at 0x243fa8b5b00>



#### AutoInt


```python
from model import AutoInt
```


```python
model = AutoInt.AutoInt(feature_fields=fields, embed_dim=16, head_num=4, attn_layers=3, mlp_dims=(32,16), dropout=0.2)
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    Train on 60000 samples, validate on 20000 samples
    60000/60000 [==============================] - 29s 477us/sample - loss: 0.4283 - auc_8: 0.7040 - val_loss: 0.4064 - val_auc_8: 0.7437
    




    <tensorflow.python.keras.callbacks.History at 0x243858b2be0>



#### FiBiNet


```python
from model import FiBiNET
```


```python
model = FiBiNET.FiBiNET(feature_fields=fields, embed_dim=8, reduction_ratio=2, pooling='mean')
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    Train on 60000 samples, validate on 20000 samples
    60000/60000 [==============================] - 75s 1ms/sample - loss: 0.4149 - auc_9: 0.7225 - val_loss: 0.4094 - val_auc_9: 0.7445
    




    <tensorflow.python.keras.callbacks.History at 0x2439817def0>



#### DCNv2


```python
from model import DCNv2
```


```python
model = DCNv2.DeepCrossNetv2(feature_fields = fields, embed_dim = 16, layer_num = 2,
                             mlp_dims = (32, 16), dropout = 0.1, cross_method = 'Matrix')
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    Train on 60000 samples, validate on 20000 samples
    60000/60000 [==============================] - 39s 645us/sample - loss: 0.4122 - auc_10: 0.7278 - val_loss: 0.4039 - val_auc_10: 0.7517
    




    <tensorflow.python.keras.callbacks.History at 0x24421bdd198>




```python
model = DCNv2.DeepCrossNetv2(feature_fields = fields, embed_dim = 16, layer_num = 2,
                             mlp_dims = (32, 16), dropout = 0.1, cross_method = 'Mix')
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    Train on 60000 samples, validate on 20000 samples
    60000/60000 [==============================] - 30s 507us/sample - loss: 0.4142 - auc_11: 0.7233 - val_loss: 0.4066 - val_auc_11: 0.7464
    




    <tensorflow.python.keras.callbacks.History at 0x24424dac080>



#### DIFM


```python
from model import DIFM
```


```python
model = DIFM.DIFM(feature_fields=fields, embed_dim=8, head_num=2, dropout=0.1)
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    1875/1875 [==============================] - 11s 6ms/step - loss: 0.4154 - auc: 0.7222 - val_loss: 0.4100 - val_auc: 0.7392
    




    <tensorflow.python.keras.callbacks.History at 0x1d2a7f34e20>



#### AFN


```python
from model import AFN
```


```python
model = AFN.AFN(feature_fields=fields, embed_size=8, hidden_size=256, dropout=0.1)
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    1875/1875 [==============================] - 14s 8ms/step - loss: 0.4203 - auc: 0.7150 - val_loss: 0.4073 - val_auc: 0.7446
    




    <tensorflow.python.keras.callbacks.History at 0x206acfd5940>



#### ONN


```python
from model import ONN
```


```python
model = ONN.ONN(feature_fields=fields, embed_dim=8, mlp_dims=[64, 32], dropout=0.1)
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=1)
```

    1875/1875 [==============================] - 137s 73ms/step - loss: 0.4377 - auc: 0.7004 - val_loss: 0.4053 - val_auc: 0.7478
    




    <tensorflow.python.keras.callbacks.History at 0x24f6a8b6a90>



## 序列模型

####  DIN

Deep Interest Net在预测的时候，对用户不同的行为的注意力是不一样的

在生成User embedding的时候，加入了Activation Unit Layer.这一层产生了每个用户行为的权重乘上相应的物品embedding，从而生产了user interest embedding的表示

实际例子： Amazon Book数据 10K

每条数据记录会有用户的行为数据

只保留了商品特征，以及历史上的商品hist的特征.


```python

# 预处理好的数据
# 处理的函数在AmazonDataPreprocress.py中
# 原始数据为.txt文件

data = pd.read_csv('../data/amazon-books-100k-preprocessed.csv', index_col = 0)
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hist_cate_0</th>
      <th>hist_cate_1</th>
      <th>hist_cate_2</th>
      <th>hist_cate_3</th>
      <th>hist_cate_4</th>
      <th>hist_cate_5</th>
      <th>hist_cate_6</th>
      <th>hist_cate_7</th>
      <th>hist_cate_8</th>
      <th>hist_cate_9</th>
      <th>...</th>
      <th>hist_cate_32</th>
      <th>hist_cate_33</th>
      <th>hist_cate_34</th>
      <th>hist_cate_35</th>
      <th>hist_cate_36</th>
      <th>hist_cate_37</th>
      <th>hist_cate_38</th>
      <th>hist_cate_39</th>
      <th>cateID</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>751</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>97</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1094</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>97</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>751</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>607</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 42 columns</p>
</div>




```python
data_X = data.iloc[:,:-1]
data_y = data.label.values
```


```python
fields = data_X.max().max()
```


```python
fields
```




    1347




```python
tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size = 0.2, random_state=42, stratify=data_y)
train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size = 0.25, random_state=42, stratify=tmp_y)
```


```python
from model import DIN
```


```python
model = DIN.DeepInterestNet(feature_dim=fields, embed_dim=8, mlp_dims=[64,32], dropout=0.2)
```


```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss = 'binary_crossentropy', metrics=[keras.metrics.AUC()])
```


```python
model.fit(train_X.values, train_y, batch_size=32, validation_data=(val_X.values, val_y), epochs=2)
```

    Train on 60000 samples, validate on 20000 samples
    Epoch 1/2
    60000/60000 [==============================] - 14s 235us/sample - loss: 0.6788 - auc: 0.5817 - val_loss: 0.6751 - val_auc: 0.5981
    Epoch 2/2
    60000/60000 [==============================] - 11s 188us/sample - loss: 0.6687 - auc: 0.6080 - val_loss: 0.6744 - val_auc: 0.5921
    




    <tensorflow.python.keras.callbacks.History at 0x1abfc62fb00>



### DIEN

相比于DIN， DIEN的改动：

1） 关注兴趣的演化过程，提出了兴趣进化网络，用序列模型做的， DIN中用户兴趣之间是相互独立的，但实际上的兴趣是不断进化的

2） 设计了一个兴趣抽取层，加入了一个二分类模型来辅助计算兴趣抽取的准确性

3） 用序列模型表达用户的兴趣动态变化性

实际的数据用例和DIN一样


```python
data = pd.read_csv('../data/amazon-books-100k-preprocessed.csv', index_col=0)
```


```python
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hist_cate_0</th>
      <th>hist_cate_1</th>
      <th>hist_cate_2</th>
      <th>hist_cate_3</th>
      <th>hist_cate_4</th>
      <th>hist_cate_5</th>
      <th>hist_cate_6</th>
      <th>hist_cate_7</th>
      <th>hist_cate_8</th>
      <th>hist_cate_9</th>
      <th>...</th>
      <th>hist_cate_32</th>
      <th>hist_cate_33</th>
      <th>hist_cate_34</th>
      <th>hist_cate_35</th>
      <th>hist_cate_36</th>
      <th>hist_cate_37</th>
      <th>hist_cate_38</th>
      <th>hist_cate_39</th>
      <th>cateID</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>751</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>97</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1094</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>97</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>99995</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>751</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>607</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>142</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 42 columns</p>
</div>




```python
fields = data.max().max()
data_X = data.iloc[:,:-1]
data_y = data.label.values
```


```python
from model.DIEN import DeepInterestEvolutionNet, auxiliary_sample
```


```python
tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size = 0.2, random_state=42, stratify=data_y)
train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size = 0.25, random_state=42, stratify=tmp_y)
```


```python
train_X_neg = auxiliary_sample(train_X)
```


```python
train_X = train_X.values
val_X = val_X.values
test_X = test_X.values
```


```python
train_loader = tf.data.Dataset.from_tensor_slices((train_X, train_X_neg, train_y)).shuffle(len(train_X)).batch(128)
```


```python
val_loader  =tf.data.Dataset.from_tensor_slices((val_X, val_y)).batch(128)
```


```python
model = DeepInterestEvolutionNet(feature_dim=fields, embed_dim=4, mlp_dims=[32,32], dropout=0.2, gru_type = 'GRU')
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
```


```python
epoches = 3
for epoch in range(epoches):
    epoch_train_loss = tf.keras.metrics.Mean()
    for batch, (x, neg_x, y) in tqdm(enumerate(train_loader)):
        with tf.GradientTape() as tape:
            out, aux_loss = model(x, neg_x)
            loss = tf.keras.losses.binary_crossentropy(y, out)
            loss = tf.reduce_mean(loss) + tf.cast(aux_loss, tf.float32)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars = zip(grads, model.trainable_variables))
        epoch_train_loss(loss)
    epoch_val_loss = tf.keras.metrics.Mean()
    for batch, (x, y) in tqdm(enumerate(val_loader)):
        out,_ = model(x)
        loss = tf.keras.losses.binary_crossentropy(y, out)
        loss = tf.reduce_mean(loss)
        epoch_val_loss(loss)
    print('EPOCH : %s, train loss : %s, val loss: %s' % (epoch,
                                                         epoch_train_loss.result().numpy(),
                                                         epoch_val_loss.result().numpy()))
```

    469it [01:42,  4.58it/s]
    157it [00:11, 14.24it/s]
    0it [00:00, ?it/s]

    EPOCH : 0, train loss : 1.9061264, val loss: 0.69325197
    

    469it [01:43,  4.55it/s]
    157it [00:11, 14.19it/s]
    0it [00:00, ?it/s]

    EPOCH : 1, train loss : 0.80915856, val loss: 0.6931492
    

    469it [01:42,  4.57it/s]
    157it [00:11, 14.26it/s]

    EPOCH : 2, train loss : 0.7702951, val loss: 0.693148
    

    
    


```python

```
