import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  GINConv,GINEConv

class LatencyModel(nn.Module):
    def __init__(self, embed_dim,edge_dim):
        super(LatencyModel, self).__init__()
        self.embed_dim = embed_dim
        self.item_embedding = torch.nn.Embedding(num_embeddings=20, embedding_dim=embed_dim)
        self.conv1 = GINEConv(nn=nn.Linear(embed_dim, 64), eps=1e-9,edge_dim = edge_dim)
        self.conv6 = GINEConv(nn=nn.Linear(64, 32), eps=1e-9, edge_dim=edge_dim)
        self.act1 = torch.nn.LeakyReLU()
        self.act2 = torch.nn.LeakyReLU()
        self.act3 = torch.nn.LeakyReLU()
        self.act4 = torch.nn.LeakyReLU()
        self.act5 = torch.nn.LeakyReLU()

    def forward(self, x, edge_index,edge_attr):
        # encode
        x = self.item_embedding(x).squeeze(1)
        #print(x.shape,edge_index.shape,edge_attr.shape)
        #print(x)
        x = self.conv1(x,edge_index,edge_attr)
        x = self.act1(x)
        x = self.conv6(x, edge_index, edge_attr)

        #x = F.relu(x)
        s_ = x @ x.T

        # return reconstructed matrices
        return s_