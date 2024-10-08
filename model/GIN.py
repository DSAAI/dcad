import torch
import torch as nn
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.nn import  GINConv

class GIN(torch.nn.Module):
     def __init__(self, hidden_dim, num_layers):
         super(GIN, self).__init__()
         self.conv1 = GINConv(mlp=nn.Sequential(nn.Linear(num_features, hidden_dim),
                         nn.ReLU(),
                         nn.Linear(hidden_dim, hidden_dim)))
         self.convs = nn.ModuleList()
         for _ in range(num_layers - 1):
             self.convs.append(GINConv(mlp=nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                     nn.ReLU(),
                     nn.Linear(hidden_dim, hidden_dim))))
         self.classify = nn.Sequential(nn.Linear(hidden_dim, num_classes))
     def forward(self, data):
         x, edge_index, BATch = data.x, data.edge_index, data.batch
         x = F.relu(self.conv1(x, edge_index))
         for conv in self.convs:
             x = F.relu(conv(x, edge_index))
         out = global_mean_pool(x, batch)
         return self.classify(out)
















