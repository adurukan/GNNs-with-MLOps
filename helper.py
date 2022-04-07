import torch
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir("data") if isfile(join("data", f))]

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

def train_test_loader(data_list, onlyfiles):
    for file_ in onlyfiles:
        G=nx.read_gpickle(f"data/{file_}")
        data = get_data_from_graph(G)
        A = get_adjacency_matrix(G)
        data.x = torch.from_numpy(A).float()
        data.y = return_labels(G)
        data.train_mask, data.val_mask, data.test_mask = retrieve_masks(data.y)
        #data.num_features = data.x.shape[1]
        data_list.append(data)
        print(data)
    print(len(data_list))
    train_dataset = data_list[len(data_list) // 10:]
    test_dataset = data_list[:len(data_list) // 10]
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10)
    return train_loader, test_loader

def get_data(onlyfiles):
    data_list = []
    for file_ in onlyfiles:
        G=nx.read_gpickle(f"data/{file_}")
        data = get_data_from_graph(G)
        A = get_adjacency_matrix(G)
        data.x = torch.from_numpy(A).float()
        data.y = return_labels(G)
        data.train_mask, data.val_mask, data.test_mask = retrieve_masks(data.y)
        data_list.append(data)
    return data_list