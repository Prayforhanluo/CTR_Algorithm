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


```python
fields = data_X.max().values # 模型输入的feature_fields
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

    EPOCH 0 train loss : 0.69772   validation loss : 0.60676   validation auc is 0.61067
    

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

    EPOCH 0 train loss : 0.57438   validation loss : 0.48955   validation auc is 0.69284
    

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

    EPOCH 0 train loss : 0.53654   validation loss : 0.48017   validation auc is 0.69726
    

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

    EPOCH 0 train loss : 0.70439   validation loss : 0.55860   validation auc is 0.63911
    

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

    EPOCH 0 train loss : 0.55712   validation loss : 0.49020   validation auc is 0.68294
    

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

    EPOCH 0 train loss : 0.56085   validation loss : 0.49273   validation auc is 0.69788
    

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

    EPOCH 0 train loss : 0.42965   validation loss : 0.40480   validation auc is 0.74818
    

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

    EPOCH 0 train loss : 0.41618   validation loss : 0.40377   validation auc is 0.74731
    

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

    EPOCH 0 train loss : 0.54381   validation loss : 0.48163   validation auc is 0.69121
    


```python

```
