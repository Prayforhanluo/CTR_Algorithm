```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow import keras
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


```python
fields = data_X.max().values # 模型输入的feature_fields
```


```python
fields
```




    array([    1,     5,     5,   986,   871,    17,   768,    61,    18,
            8543, 47308,  2605,     3,     3,   447,     4,     5,   140,
               3,    37,   143,    32], dtype=int64)




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

    1875/1875 [==============================] - 2s 1ms/step - loss: 0.4248 - auc: 0.6999 - val_loss: 0.4139 - val_auc: 0.7286
    




    <tensorflow.python.keras.callbacks.History at 0x1395e7092b0>



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

    1875/1875 [==============================] - 8s 4ms/step - loss: 0.4172 - auc_1: 0.7168 - val_loss: 0.4071 - val_auc_1: 0.7449
    




    <tensorflow.python.keras.callbacks.History at 0x1395fce5070>



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

    1875/1875 [==============================] - 132s 70ms/step - loss: 0.4077 - auc_2: 0.7369 - val_loss: 0.4032 - val_auc_2: 0.7527
    




    <tensorflow.python.keras.callbacks.History at 0x1395fd41280>



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

    1875/1875 [==============================] - 10s 5ms/step - loss: 0.4247 - auc_3: 0.7009 - val_loss: 0.4108 - val_auc_3: 0.7329
    




    <tensorflow.python.keras.callbacks.History at 0x13904f56700>



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

    1875/1875 [==============================] - 9s 5ms/step - loss: 0.4216 - auc_4: 0.7084 - val_loss: 0.4061 - val_auc_4: 0.7436
    




    <tensorflow.python.keras.callbacks.History at 0x13905d387f0>



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

    1875/1875 [==============================] - 12s 6ms/step - loss: 0.4259 - auc_5: 0.7008 - val_loss: 0.4083 - val_auc_5: 0.7396
    




    <tensorflow.python.keras.callbacks.History at 0x13909067940>



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

    1875/1875 [==============================] - 9s 5ms/step - loss: 0.4718 - auc_6: 0.6593 - val_loss: 0.4095 - val_auc_6: 0.7360
    




    <tensorflow.python.keras.callbacks.History at 0x13909f55880>



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

    1875/1875 [==============================] - 9s 5ms/step - loss: 0.4173 - auc_7: 0.7175 - val_loss: 0.4051 - val_auc_7: 0.7464
    




    <tensorflow.python.keras.callbacks.History at 0x13909cf5370>



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
    60000/60000 [==============================] - 28s 472us/sample - loss: 0.4216 - auc: 0.7077 - val_loss: 0.4071 - val_auc: 0.7415
    




    <tensorflow.python.keras.callbacks.History at 0x2103e456fd0>




```python

```
