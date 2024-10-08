import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  GINConv,GINEConv,RGATConv

class CombineModel(nn.Module):

    def __init__(self, embed_dim,edge_dim,num_realations):
        super(CombineModel, self).__init__()
        self.embed_dim = embed_dim
        self.item_embedding = torch.nn.Embedding(num_embeddings=20, embedding_dim=embed_dim)
        self.conv1 = RGATConv(embed_dim, 16,num_realations,edge_dim = edge_dim)#GINEConv(nn=nn.Linear(embed_dim, 64), eps=1e-9, edge_dim=edge_dim)
        self.act1 = torch.nn.LeakyReLU()
        self.sturctconv1 = RGATConv(16, 32,num_realations,edge_dim = edge_dim)
        self.latencyconv1 = RGATConv(16, 32,num_realations,edge_dim = edge_dim)


    def forward(self,x, edge_index,edge_attr,edge_type):
        embedding_x = self.item_embedding(x).squeeze(1)
        x = self.conv1(embedding_x, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        h = self.act1(x)


        #print(h.shape,edge_index.shape ,edge_type.shape,edge_attr.shape)
        h_ = self.sturctconv1(h, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        #print(h_.shape)

        s_ = h_ @ h_.T

        x_ = self.latencyconv1(h, edge_index, edge_type=edge_type, edge_attr=edge_attr)


        return embedding_x, x_ , s_



'''
class CombineModel(nn.Module):

    def __init__(self, embed_dim,edge_dim,num_realations):
        super(CombineModel, self).__init__()
        self.embed_dim = embed_dim
        self.item_embedding = torch.nn.Embedding(num_embeddings=20, embedding_dim=embed_dim)
        self.conv1 = RGATConv(embed_dim, 64,num_realations,edge_dim = edge_dim)#GINEConv(nn=nn.Linear(embed_dim, 64), eps=1e-9, edge_dim=edge_dim)
        self.act1 = torch.nn.LeakyReLU()
        self.conv2 = RGATConv(64, 32,num_realations,edge_dim = edge_dim)
        self.act2 = torch.nn.LeakyReLU()
        self.conv3 = RGATConv(32, 16,num_realations,edge_dim = edge_dim)
        self.act3 = torch.nn.LeakyReLU()

        self.sturctconv1 = RGATConv(16, 32,num_realations,edge_dim = edge_dim)
        self.sturctact1 = torch.nn.LeakyReLU()
        self.sturctconv2 = RGATConv(32, 64,num_realations,edge_dim = edge_dim)
        self.sturctact2 = torch.nn.LeakyReLU()
        self.sturctconv3 = RGATConv(64, 128, num_realations, edge_dim=edge_dim)


        self.latencyconv1 = RGATConv(16, 32,num_realations,edge_dim = edge_dim)
        self.latencyact1 = torch.nn.LeakyReLU()
        self.latencyconv2 = RGATConv(32, 64, num_realations, edge_dim=edge_dim)
        self.latencyact2 = torch.nn.LeakyReLU()
        self.latencyconv3 = RGATConv(64, 128, num_realations, edge_dim=edge_dim)



    def forward(self,x, edge_index,edge_attr,edge_type):
        embedding_x = self.item_embedding(x).squeeze(1)
        x = self.conv1(embedding_x, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        x = self.act1(x)
        x = self.conv2(x, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        x = self.act2(x)
        x = self.conv3(x, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        h = self.act3(x)

        #print(h.shape,edge_index.shape ,edge_type.shape,edge_attr.shape)
        h_ = self.sturctconv1(h, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        #print(h_.shape)
        h_ = self.sturctact1(h_)
        h_ = self.sturctconv2(h_, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        h_ = self.sturctact2(h_)
        h_ = self.sturctconv3(h_, edge_index, edge_type=edge_type, edge_attr=edge_attr)

        s_ = h_ @ h_.T

        x_ = self.latencyconv1(h, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        x_ = self.latencyact1(x_)
        x_ = self.latencyconv2(x_, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        x_ = self.latencyact2(x_)
        x_ = self.latencyconv3(x_, edge_index, edge_type=edge_type, edge_attr=edge_attr)

        return embedding_x,x_ , s_


'''












