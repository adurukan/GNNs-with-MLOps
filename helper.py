import torch
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import networkx as nx
import os
from os import listdir
from os.path import isfile, join
import json
import glob

graph_data = [f for f in listdir("train_data") if isfile(join("train_data", f))]
test_data = [f for f in listdir("test_data") if isfile(join("test_data", f))]

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

def train_test_loader(data_list, graph_data):
    for file_ in graph_data:
        G=nx.read_gpickle(f"train_data/{file_}")
        data = get_data_from_graph(G)
        A = get_adjacency_matrix(G)
        data.x = torch.from_numpy(A).float()
        data.y = return_labels(G)
        data.train_mask, data.val_mask, data.test_mask = retrieve_masks(data.y)
        #data.num_features = data.x.shape[1]
        data_list.append(data)
    train_dataset = data_list[len(data_list) // 10:]
    test_dataset = data_list[:len(data_list) // 10]
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10)
    return train_loader, test_loader

def get_data(graph_data, folder_path):
    data_list = []
    for file_ in graph_data:
        if 'gpickle' in file_:
            #print(file_)
            G=nx.read_gpickle(f"{folder_path}/{file_}")
            #print(G)
            data = get_data_from_graph(G)
            A = get_adjacency_matrix(G)
            data.x = torch.from_numpy(A).float()
            data.y = return_labels(G)
            data.train_mask, data.val_mask, data.test_mask = retrieve_masks(data.y)
            data_list.append(data)
    return data_list

def report_training_accuracy(accuracy_dict):
    metrics = [f for f in listdir("metrics") if isfile(join("metrics", f))]
    if metrics == []:
        with open(f"metrics/0.json", "w") as outfile:
            json.dump(accuracy_dict, outfile, indent = 8)
    else:
        new_file_name = str(max(list(map(lambda s: int(s.split('.')[0]), metrics))) + 1) + '.json' # ['1.json', '3.json', '2.json'] -> '4.json'
        with open(f"metrics/{new_file_name}", "w") as outfile:
            json.dump(accuracy_dict, outfile, indent = 8)
