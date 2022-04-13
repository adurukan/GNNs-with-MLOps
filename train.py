import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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
from os import listdir
from os.path import isfile, join
import os.path as osp
from torch_geometric.datasets import TUDataset
from gat_net import GAT
from helper import get_data, graph_data
"""
In this script, model&data will be loaded and model will be trained accordingly.
"""
with open("logger.txt", "w") as outfile:
    outfile.write("train.py -> Imports are successful.")

#onlyfiles = [f for f in listdir("data") if isfile(join("data", f))]
#onlyfiles = get_data(graph_data, "train_data")
def return_labels(G):
    nodes = G.nodes
    labels = nx.get_node_attributes(G,"suspicious")
    y = []
    for label in labels.values():
        y.append(label)
    y = np.asarray(y)
    y = torch.from_numpy(y)
    y = y.type(torch.long)
    return y

def get_adjacency_matrix(G):
    A = nx.adjacency_matrix(G)
    A = A.todense()
    A = np.asarray(A)
    return A

def get_data_from_graph(G):
    data = from_networkx(G)
    return data

def retrieve_masks(y):
    train_mask = torch.full_like(y, False, dtype=bool)
    train_mask[:] = True
    val_mask = torch.full_like(y, False, dtype=bool)
    val_mask[:] = False
    test_mask = torch.full_like(y, False, dtype=bool)
    test_mask[:] = False
    return train_mask, val_mask, test_mask

# def train_test_loader(data_list, onlyfiles):
#     for file_ in onlyfiles:
#         G=nx.read_gpickle(f"data/{file_}")
#         data = get_data_from_graph(G)
#         A = get_adjacency_matrix(G)
#         data.x = torch.from_numpy(A).float()
#         #print(f"shape of data.x: {data.x.shape}")
#         data.y = return_labels(G)
#         #data.train_mask, data.val_mask, data.test_mask = retrieve_masks(data.y)
#         data.num_features = data.x.shape[1]
#         #print(data.num_features)
#         data_list.append(data)
#     #print(f"length of data_list: {len(data_list)}")
#     train_dataset = data_list[len(data_list) // 10:]
#     test_dataset = data_list[:len(data_list) // 10]
#     train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=10)
#     return train_loader, test_loader

@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask'):
        acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
        accs.append(acc)
    return accs

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    return loss


if __name__ == "__main__":
    data_list = get_data(graph_data, "train_data")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load("models/gat_300_2")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for data in data_list:
        data = data.to(device)
        for epoch in range(1, 20):
            loss = train(data)
            train_acc = test(data)
        print(f"loss: {loss}, train_ accuracy : {train_acc}")
            #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f}')
    torch.save(model, "models/gat_300_2")
    