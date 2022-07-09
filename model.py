import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class Graph(torch.nn.Module):
    def __init__(self, data, output_size=8):
        super(Graph, self).__init__()
        self.x, self.edge_index, self.device = data.x, data.edge_index, data.x.device
        input_size = data.x.shape[1]
        hidden = int(input_size/2)
        self.sage1 = SAGEConv(input_size, hidden)
        self.sage2 = SAGEConv(hidden, output_size)
        self.embeddings = torch.zeros((self.x.shape[0], output_size), device=self.device)

    def forward(self, x):
        if self.training:
            out = self.sage1(self.x, self.edge_index)
            out = F.relu(out)
            out = F.dropout(out, training=self.training)
            self.embeddings = self.sage2(out, self.edge_index)

        return self.embeddings[x]


class GraphAE(torch.nn.Module):
    def __init__(self, rating_mat, data, embedding_size, dropout=0.0, method='add'):
        super(GraphAE, self).__init__()
        N, M = rating_mat.shape
        self.rating_mat, self.embedding_size, self.method = rating_mat, embedding_size, method
        self.embedding_user = nn.Embedding(N, embedding_size)
        
        self.graph = Graph(data, embedding_size)
        
        self.encoder = nn.Sequential(
            nn.Linear(M, embedding_size), 
            nn.Dropout(dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, M),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        items = self.rating_mat[x] 
        embedding_ae = self.encoder(items)
        embedding_user = self.embedding_user(x)
        
        embedding_graph = self.graph(x)
        if self.method == 'add':
            out = embedding_graph + embedding_ae
        elif self.method == 'mm':
            out = embedding_graph * embedding_ae

        out = self.sigmoid(out)
            
        out = self.decoder(out)
            
        return out