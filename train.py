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
from gat_2 import GAT
"""
In this script, model&data will be loaded and model will be trained accordingly.
"""
with open("logger.txt", "w") as outfile:
    outfile.write("train.py -> Imports are successful.")

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
        print(f"shape of data.x: {data.x.shape}")
        data.y = return_labels(G)
        #data.train_mask, data.val_mask, data.test_mask = retrieve_masks(data.y)
        data.num_features = data.x.shape[1]
        #print(data.num_features)
        data_list.append(data)
    #print(f"length of data_list: {len(data_list)}")
    train_dataset = data_list[len(data_list) // 10:]
    test_dataset = data_list[:len(data_list) // 10]
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10)
    return train_loader, test_loader

@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    j = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        #print(f"type of out: {type(out.argmax(-1))} shape: {out.argmax(-1).shape}")
        #print(f"type of data.y: {type(data.y)} shape: {data.y.shape}")
        probs = torch.softmax(out, dim=1)
        winners = probs.argmax(dim=1)
        corrects = (winners == data.y)
        accuracy = corrects.sum().float() / float( data.y.size(0) )
        #print(f"shape of accuracy: {accuracy}")
        total_correct += accuracy
        j = j + 1
        #total_correct += int((out.argmax(-1) == data.y).sum())
    #total_correct = total_correct / int(data.y.shape[0])
    #print(len(loader.dataset))
    #print(j)
    return total_correct / j
    #return total_correct / len(loader.dataset)

def train(train_loader):
    model.train()
    total_loss = 0
    for data in train_loader:
        # print(data)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)
    # file_ = onlyfiles[0]
    # G=nx.read_gpickle(f"data/{file_}")
    # data = get_data_from_graph(G)
    # data_list.append(data)
    # A = get_adjacency_matrix(G)
    # data.x = torch.from_numpy(A).float()
    # data.y = return_labels(G)
    # data.train_mask, data.val_mask, data.test_mask = retrieve_masks(data.y)
    # data.num_features = data.x.shape[1]
    # # print(f"data: {data}")
    # model = GAT(data.num_features)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # model.train()
    # for epoch in range(100):
    #     model.train()
    #     optimizer.zero_grad()
    #     out = model(data)
    #     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        
    #     if epoch%10 == 0:
    #         print(loss)
        
    #     loss.backward()
    #     optimizer.step()

    # model.eval()
    # torch.save(model, "models/gat_2")

    # for file_ in onlyfiles[1:]:
    #     G=nx.read_gpickle(f"data/{file_}")
    #     data = get_data_from_graph(G)
    #     data_list.append(data)
    #     print(type(data))

    # train_dataset = data_list[len(data_list) // 10:]
    # test_dataset = data_list[:len(data_list) // 10]
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=128)
    # #loader = DataLoader(data_list, batch_size=32)
    # print(train_loader)
    # print(test_loader)
    #     # A = get_adjacency_matrix(G)
    #     # data.x = torch.from_numpy(A).float()
    #     # data.y = return_labels(G)
    #     # data.train_mask, data.val_mask, data.test_mask = retrieve_masks(data.y)
    #     # data.num_features = data.x.shape[1]
    #     # # print(f"type of y: {type(data.y)} and length of y: {data.y.shape}")
    #     # # print(f"type of x: {type(data.x)} and length of x: {data.x.shape}")
    #     # # print(f"type of num_features: {type(data.num_features)} and length of num_features: {data.num_features}")
    #     # # print(f"type of train_mask: {type(data.train_mask)} and length of train_mask: {data.train_mask.shape}")
    #     # # print(f"type of val_mask: {type(data.val_mask)} and length of val_mask: {data.val_mask.shape}")
    #     # # print(f"type of test_mask: {type(data.test_mask)} and length of test_mask: {data.test_mask.shape}")
    #     # print(f"data: {data}")

    #     # model = GAT(data.num_features)
    #     # model = torch.load("models/gat_2")
    #     # print(f"model: {model}")
    #     # optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    #     # model.train()
    #     # for epoch in range(100):
    #     #     model.train()
    #     #     optimizer.zero_grad()
    #     #     out = model(data)
    #     #     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            
    #     #     if epoch%10 == 0:
    #     #         print(loss)
            
    #     #     loss.backward()
    #     #     optimizer.step()

    #     # model.eval()
    #     # torch.save(model, "models/gat_2")

if __name__ == "__main__":
    data_list = []
    train_loader, test_loader = train_test_loader(data_list, onlyfiles)
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
    # dataset = TUDataset(path, name='MUTAG').shuffle()
    # train_dataset = dataset[len(dataset) // 10:]
    # test_dataset = dataset[:len(dataset) // 10]
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(300).to(device)
    # torch.save(model, "models/gat_2")
    #model = torch.load("models/gat_2")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    print(f"model: {model}")
    for epoch in range(1, 20):
        loss = train(train_loader)
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f} '
            f'Test Acc: {test_acc:.4f}')
    #torch.save(model, "models/gat_2")
    # ONLY save the model optimized parameters & state as checkpoints, 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
    # Reload for further training:
#     checkpoint = torch.load(PATH)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']

#     model.train()
