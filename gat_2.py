import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import from_networkx
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
import matplotlib.pyplot as plt
from random import randint
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
from random import randint
import matplotlib.pyplot as plt
import os

# G=nx.read_gpickle("data/dataset_0_D.gpickle")
# diamond = nx.read_gpickle("diamonds/diamond_0_D.gpickle")
# A = nx.adjacency_matrix(G)
# A = A.todense()
# A = np.asarray(A)
# data = from_networkx(G)
# #diamond = from_networkx(diamond)
# n = int(data.num_nodes)
# print(f"Number of Nodes: {n}")
# print(f"Number of Diamond Nodes: {diamond.nodes}")
# data.x = torch.from_numpy(A).float()
# y = np.zeros(n)
# y[diamond.nodes] = 1
# data.y = torch.from_numpy(y)
# data.y = data.y.type(torch.long)
# data.num_classes = torch.unique(data.y).size(dim=0)

class GAT(torch.nn.Module):
    def __init__(self, num_features):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        self.num_features = num_features # normally comes from dataset
        self.num_classes = 2 # normally comes from dataset

        self.conv1 = GATConv(self.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, self.num_classes, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = GAT().to(device)
# model = GAT(376)
# torch.save(model.state_dict(), "models/gat_2_state_dict")