```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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
#train, validation, test 集合
tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size = 0.2, random_state=42, stratify=data_y)
train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size = 0.25, random_state=42, stratify=tmp_y)


# 数据量小, 可以直接读
train_X = torch.from_numpy(train_X.values).long()
val_X = torch.from_numpy(val_X.values).long()
test_X = torch.from_numpy(test_X.values).long()

train_y = torch.from_numpy(train_y).long()
val_y = torch.from_numpy(val_y).long()
test_y = torch.from_numpy(test_y).long()

train_set = Data.TensorDataset(train_X, train_y)
val_set = Data.TensorDataset(val_X, val_y)
train_loader = Data.DataLoader(dataset=train_set,
                               batch_size=32,
                               shuffle=True)
val_loader = Data.DataLoader(dataset=val_set,
                             batch_size=32,
                             shuffle=False)
```

## 训练过程
   
    数据是avazu数据的随机10万条
    优化器统一Adam， lr = 0.001
    epoch 为 1, batch_size = 32
    主要的目的是跑通所有的模型
    epoch多几次, 调调参数对稍微复杂的网络有好处
    
    
    tips : 类别特征embedding等价于一层没有bias项的全连接，所以模型中几乎都用embedding来模拟LR线性过程


```python
epoches = 1
```


```python
def train(model):
    for epoch in range(epoches):
        train_loss = []
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        model.train()
        for batch, (x, y) in enumerate(train_loader):
            pred = model(x)
            loss = criterion(pred, y.float().detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        val_loss = []
        prediction = []
        y_true = []
        with torch.no_grad():
            for batch, (x, y) in enumerate(val_loader):
                pred = model(x)
                loss = criterion(pred, y.float().detach())
                val_loss.append(loss.item())
                prediction.extend(pred.tolist())
                y_true.extend(y.tolist())
        val_auc = roc_auc_score(y_true=y_true, y_score=prediction)
        print ("EPOCH %s train loss : %.5f   validation loss : %.5f   validation auc is %.5f" % (epoch, np.mean(train_loss), np.mean(val_loss), val_auc))        
    return train_loss, val_loss, val_auc
```

#### LR


```python
from model import LR
```


```python
model = LR.LogisticRegression(feature_fields=fields)
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.76449   validation loss : 0.64623   validation auc is 0.59039
    

#### FM


```python
from model import FM
```


```python
model = FM.FactorizationMachine(feature_fields=fields, embed_dim=8)
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.60432   validation loss : 0.49426   validation auc is 0.67547
    

#### FFM


```python
from model import FFM
```


```python
model = FFM.FieldAwareFactorizationMachine(field_dims=fields, embed_dim=8)
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.53437   validation loss : 0.48431   validation auc is 0.69762
    

#### AFM


```python
from model import AFM
```


```python
model = AFM.AttentionalFactorizationMachine(feature_fields=fields, embed_dim=8, attn_size=8, dropouts=(0.25, 0.25))
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.73301   validation loss : 0.57867   validation auc is 0.64243
    

#### DeepFM


```python
from model import DeepFM
```


```python
model = DeepFM.DeepFM(feature_fields=fields, embed_dim=8, mlp_dims=(32,16), dropout=0.2)
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.97898   validation loss : 0.50353   validation auc is 0.68853
    

#### xDeepFM


```python
from model import xDeepFM
```


```python
model = xDeepFM.xDeepFM(feature_fields=fields, embed_dim=8, mlp_dims=(32,16), 
                        dropout=0.2, cross_layer_sizes=(16,16),split_half=False)
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.58850   validation loss : 0.49531   validation auc is 0.68098
    

#### PNN


```python
from model import PNN
```


```python
model = PNN.PNN(feature_fields=fields, embed_dim=8, mlp_dims=(32, 16), dropout=0.2)
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.44456   validation loss : 0.40605   validation auc is 0.74571
    

#### DCN


```python
from model import DCN
```


```python
model = DCN.DeepCrossNet(feature_fields=fields, embed_dim=8, num_layers=3, mlp_dims=(16,16), dropout=0.2)
```


```python
_ = train(model)
```

    C:\Users\jiguang\anaconda3\envs\py36\lib\site-packages\torch\nn\modules\container.py:435: UserWarning: Setting attributes on ParameterList is not supported.
      warnings.warn("Setting attributes on ParameterList is not supported.")
    

    EPOCH 0 train loss : 0.41461   validation loss : 0.40514   validation auc is 0.75012
    

#### AutoInt


```python
from model import AutoInt
```


```python
model = AutoInt.AutoIntNet(feature_fields=fields, embed_dim=8, head_num = 2, 
                           attn_layers=3, mlp_dims=(32, 16), dropout=0.2)
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.57498   validation loss : 0.50322   validation auc is 0.68955
    

#### FiBiNet


```python
from model import FiBiNET
```


```python
model = FiBiNET.FiBiNET(feature_fields=fields, embed_dim=8, reduction_ratio=2, pooling='mean')
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.41589   validation loss : 0.40396   validation auc is 0.74876
    

#### DCNv2


```python
from model import DCNv2
```


```python
model =  DCNv2.DeepCrossNetv2(feature_fields = fields, embed_dim = 16, layer_num = 2, mlp_dims = (32, 16), dropout = 0.1)
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.41441   validation loss : 0.40312   validation auc is 0.74947
    


```python
model =  DCNv2.DeepCrossNetv2(feature_fields = fields, embed_dim = 16, layer_num = 2, mlp_dims = (32, 16), dropout = 0.1,cross_method='Matrix')
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.41366   validation loss : 0.40360   validation auc is 0.75210
    

#### DIFM


```python
from model import DIFM
```


```python
model = DIFM.DIFM(feature_fields=fields, embed_size=8, head_num=2, dropout=0.1)
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.42742   validation loss : 0.41314   validation auc is 0.73831
    

#### AFN


```python
from model import AFN
```


```python
model = AFN.AFN(feature_fields=fields, embed_size=8, hidden_size=256, dropout=0.1)
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.55026   validation loss : 0.51018   validation auc is 0.68282
    

## 序列模型

#### DIN

Deep Interest Net在预测的时候，对用户不同的行为的注意力是不一样的

在生成User embedding的时候，加入了Activation Unit Layer.这一层产生了每个用户行为的权重乘上相应的物品embedding，从而生产了user interest embedding的表示

实际例子： Amazon Book数据 10K

每条数据记录会有用户的行为数据

只保留了商品特征，以及历史上的商品hist的特征.


```python
data = pd.read_csv('../data/amazon-books-100k.txt')
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
      <th>label</th>
      <th>userID</th>
      <th>itemID</th>
      <th>cateID</th>
      <th>hist_item_list</th>
      <th>hist_cate_list</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>AZPJ9LUT0FEPY</td>
      <td>B00AMNNTIA</td>
      <td>Literature &amp; Fiction</td>
      <td>0307744434|0062248391|0470530707|0978924622|15...</td>
      <td>Books|Books|Books|Books|Books</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>AZPJ9LUT0FEPY</td>
      <td>0800731603</td>
      <td>Books</td>
      <td>0307744434|0062248391|0470530707|0978924622|15...</td>
      <td>Books|Books|Books|Books|Books</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>A2NRV79GKAU726</td>
      <td>B003NNV10O</td>
      <td>Russian</td>
      <td>0814472869|0071462074|1583942300|0812538366|B0...</td>
      <td>Books|Books|Books|Books|Baking|Books|Books</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>A2NRV79GKAU726</td>
      <td>B000UWJ91O</td>
      <td>Books</td>
      <td>0814472869|0071462074|1583942300|0812538366|B0...</td>
      <td>Books|Books|Books|Books|Baking|Books|Books</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>A2GEQVDX2LL4V3</td>
      <td>0321334094</td>
      <td>Books</td>
      <td>0743596870|0374280991|1439140634|0976475731</td>
      <td>Books|Books|Books|Books</td>
    </tr>
    <tr>
      <th>...</th>
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
      <td>A3I7LS4H993CXB</td>
      <td>1481872060</td>
      <td>Books</td>
      <td>1936826135|1250014409|1480219851|1484823664|14...</td>
      <td>Books|Books|Books|Books|Books|Literature &amp; Fic...</td>
    </tr>
    <tr>
      <th>99996</th>
      <td>0</td>
      <td>AP00RAQ20KM12</td>
      <td>1414334095</td>
      <td>Books</td>
      <td>0312328796|0758207182|0739470140|1601621450|18...</td>
      <td>Books|Books|Books|Books|Books|Books|Books|Book...</td>
    </tr>
    <tr>
      <th>99997</th>
      <td>1</td>
      <td>AP00RAQ20KM12</td>
      <td>B0063LINHW</td>
      <td>Historical</td>
      <td>0312328796|0758207182|0739470140|1601621450|18...</td>
      <td>Books|Books|Books|Books|Books|Books|Books|Book...</td>
    </tr>
    <tr>
      <th>99998</th>
      <td>0</td>
      <td>A1ZVJYANTLTLVP</td>
      <td>0762419229</td>
      <td>Books</td>
      <td>0743470117|0395851580|1451661215|0312342020</td>
      <td>Books|Books|Books|Books</td>
    </tr>
    <tr>
      <th>99999</th>
      <td>1</td>
      <td>A1ZVJYANTLTLVP</td>
      <td>1455507202</td>
      <td>Books</td>
      <td>0743470117|0395851580|1451661215|0312342020</td>
      <td>Books|Books|Books|Books</td>
    </tr>
  </tbody>
</table>
<p>100000 rows × 6 columns</p>
</div>




```python
# AmazonBookPreprocess Function comes from ../data/AmazonDataPreprocess.py
data = AmazonBookPreprocess(data)
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
```


```python
fields
```




    1347




```python
data_X = data.iloc[:,:-1]
data_y = data.label.values
```


```python
#train, validation, test 集合
tmp_X, test_X, tmp_y, test_y = train_test_split(data_X, data_y, test_size = 0.2, random_state=42, stratify=data_y)
train_X, val_X, train_y, val_y = train_test_split(tmp_X, tmp_y, test_size = 0.25, random_state=42, stratify=tmp_y)


# 数据量小, 可以直接读
train_X = torch.from_numpy(train_X.values).long()
val_X = torch.from_numpy(val_X.values).long()
test_X = torch.from_numpy(test_X.values).long()

train_y = torch.from_numpy(train_y).long()
val_y = torch.from_numpy(val_y).long()
test_y = torch.from_numpy(test_y).long()

train_set = Data.TensorDataset(train_X, train_y)
val_set = Data.TensorDataset(val_X, val_y)
train_loader = Data.DataLoader(dataset=train_set,
                               batch_size=32,
                               shuffle=True)
val_loader = Data.DataLoader(dataset=val_set,
                             batch_size=32,
                             shuffle=False)
```


```python
from model import DIN
```


```python
model = DIN.DeepInterestNet(feature_dim=fields, embed_dim=8, mlp_dims=[64,32], dropout=0.2)
```


```python
epoches = 1
```


```python
_ = train(model)
```

    EPOCH 0 train loss : 0.68228   validation loss : 0.67945   validation auc is 0.58338
    

#### DIEN

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
```


```python
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
# 数据量小, 可以直接读
train_X = torch.from_numpy(train_X.values).long()
train_X_neg = torch.from_numpy(train_X_neg).long()
val_X = torch.from_numpy(val_X.values).long()
test_X = torch.from_numpy(test_X.values).long()

train_y = torch.from_numpy(train_y).long()
val_y = torch.from_numpy(val_y).long()
test_y = torch.from_numpy(test_y).long()

train_set = Data.TensorDataset(train_X, train_X_neg, train_y)
val_set = Data.TensorDataset(val_X, val_y)
train_loader = Data.DataLoader(dataset=train_set,
                               batch_size=32,
                               shuffle=True)
val_loader = Data.DataLoader(dataset=val_set,
                             batch_size=32,
                             shuffle=False)
```


```python
def train_dien(model):
    for epoch in range(epoches):
        train_loss = []
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        model.train()
        for batch, (x, neg_x, y) in enumerate(train_loader):
            pred, auxiliary_loss = model(x, neg_x)
            loss = criterion(pred, y.float().detach()) + auxiliary_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        val_loss = []
        prediction = []
        y_true = []
        with torch.no_grad():
            for batch, (x, y) in enumerate(val_loader):
                pred, _ = model(x)
                loss = criterion(pred, y.float().detach())
                val_loss.append(loss.item())
                prediction.extend(pred.tolist())
                y_true.extend(y.tolist())
        val_auc = roc_auc_score(y_true=y_true, y_score=prediction)
        print ("EPOCH %s train loss : %.5f   validation loss : %.5f   validation auc is %.5f" % (epoch, np.mean(train_loss), np.mean(val_loss), val_auc))        
    return train_loss, val_loss, val_auc
```


```python
dien = DeepInterestEvolutionNet(feature_dim=fields, embed_dim=4, hidden_size=4, mlp_dims=[64,32], dropout=0.2)
```


```python
_ = train_dien(dien)
```

    EPOCH 0 train loss : 0.68810   validation loss : 0.68992   validation auc is 0.58069
    EPOCH 1 train loss : 0.68111   validation loss : 0.70098   validation auc is 0.58106
    


```python

```
