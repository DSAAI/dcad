import torch
import torch.nn as nn
from torch_geometric.nn import  RGATConv,GATConv

class StructureModel(nn.Module):
    def __init__(self,embed_dim,edge_dim):
        super(StructureModel, self).__init__()
        self.embed_dim = embed_dim
        self.edge_dim = edge_dim
        self.num_realations = 1
        self.item_embedding = torch.nn.Embedding(num_embeddings=10000, embedding_dim=embed_dim)
        self.shared_encoder1 = RGATConv(embed_dim, 128,self.num_realations,edge_dim = self.edge_dim)
        self.act1 = torch.nn.LeakyReLU()
        self.shared_encoder2 = RGATConv(128, 64,self.num_realations,edge_dim = self.edge_dim)
        self.act2 = torch.nn.LeakyReLU()
        self.shared_encoder3 = RGATConv(64, 32,self.num_realations,edge_dim = self.edge_dim)
        self.act3 = torch.nn.LeakyReLU()
        self.struct_decoder1 = RGATConv(32, 64,self.num_realations,edge_dim = self.edge_dim)
        self.act4 = torch.nn.LeakyReLU()
        self.struct_decoder2 = RGATConv(64, 128,self.num_realations,edge_dim = self.edge_dim)

    def forward(self,x,edge_index,edge_attr,edge_type):
        # encode torch.tensor([0 for i in edge_attr], dtype=torch.long)
        x = self.item_embedding(x).squeeze(1)
        #print(len(edge_attr),len(edge_attr[0]))
        x = self.shared_encoder1(x, edge_index, edge_type=edge_type,edge_attr = edge_attr)
        x = self.act1(x)
        #print("aaa")
        #print(len(x),len(edge_index[0]))

        
        x = self.shared_encoder2(x, edge_index, edge_type=edge_type,edge_attr = edge_attr)
        x = self.act2(x)
        h = self.shared_encoder3(x, edge_index, edge_type=edge_type,edge_attr = edge_attr)
        h = self.act3(h)
        h = self.struct_decoder1(h, edge_index, edge_type=edge_type,edge_attr = edge_attr)
        h = self.act4(h)
        h_ = self.struct_decoder2(h, edge_index, edge_type=edge_type,edge_attr = edge_attr)


        # decode adjacency matrix
        s_ = h_ @ h_.T
        # return reconstructed matrices
        return s_
