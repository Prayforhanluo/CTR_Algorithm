import numpy as np
import torch
import torch.nn as nn

class CIN(nn.Module):
    """
        Compressed Interaction Network.
    """
    def __init__(self, input_dim, cross_layer_sizes, split_half = True):
        super(CIN, self).__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                              stride = 1, dilation = 1, bias = True))
            
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size = cross_layer_size // 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        
        self.fc = nn.Linear(fc_input_dim, 1)
        
    def forward(self, x):
        """
            x : (batch_size, num_fields, embed_dim)
        """
        xs = []
        x0, h_ = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h_.unsqueeze(1)     ## Z(k+1)
            batch_size, f0_dim, f1_dim, embed_dim = x.shape
            x = x.contiguous().view(batch_size, f0_dim * f1_dim, embed_dim)
            x = self.conv_layers[i](x)
            x = nn.functional.relu(x)
            
            if self.split_half and i != self.num_layers - 1:
                x, h_ = torch.split(x, x.shape[1] // 2, dim = 1)
            else:
                h_ = x
            xs.append(x)
        x = torch.cat(xs, dim = 1)
        x = torch.sum(x, dim = 2)
        x = self.fc(x)
        
        return x
    
class xDeepFM(nn.Module):
    """
        xDeepFM
    """
    def __init__(self, feature_fields, embed_dim, mlp_dims, dropout, cross_layer_sizes, split_half = True):
        super().__init__()
        self.feature_fields = feature_fields
        self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        
        # Embedding layer
        self.embedding = nn.Embedding(sum(feature_fields)+1, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        self.embedding_out_dim = len(feature_fields) * embed_dim
        
        #DNN layer
        dnn_layers = []
        input_dim = self.embedding_out_dim
        self.mlp_dims = mlp_dims
        for mlp_dim in mlp_dims:
            # 全连接层
            dnn_layers.append(nn.Linear(input_dim, mlp_dim))
            dnn_layers.append(nn.BatchNorm1d(mlp_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(p = dropout))
            input_dim = mlp_dim
        dnn_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*dnn_layers)      
    
        # CIN layer
        self.CIN = CIN(len(feature_fields), cross_layer_sizes=cross_layer_sizes, split_half=split_half)
        
        # Linear layer
        self.linear = torch.nn.Embedding(sum(feature_fields)+1, 1)
        self.bias = torch.nn.Parameter(torch.zeros((1,)))
    
    def forward(self, x):
        tmp = x + x.new_tensor(self.offsets).unsqueeze(0)
        
        embeded_x = self.embedding(tmp)
        
        # Fix: Sum over the field dimension to make linear_part have shape (batch_size, 1)
        linear_part = self.linear(tmp).sum(dim=1) + self.bias
        CIN_part = self.CIN(embeded_x)
        mlp_part = self.mlp(embeded_x.view(-1, self.embedding_out_dim))
        
        # Ensure all parts have the same shape (batch_size, 1)
        x = linear_part + CIN_part + mlp_part
        x = torch.sigmoid(x)
        
        return x