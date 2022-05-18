import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

# from torch_geometric.nn import GATConv
from gat_net import GAT
import networkx as nx
from helper import get_data, graph_data

"""
In this script, model&data will be loaded and model will be trained accordingly.
"""
with open("logger.txt", "w") as outfile:
    outfile.write("train.py -> Imports are successful.")

# onlyfiles = [f for f in listdir("data") if isfile(join("data", f))]
# onlyfiles = get_data(graph_data, "train_data")
def return_labels(G):
    """Returning labels for training."""
    nodes = G.nodes
    labels = nx.get_node_attributes(G, "suspicious")
    y = []
    for label in labels.values():
        y.append(label)
    y = np.asarray(y)
    y = torch.from_numpy(y)
    y = y.type(torch.long)
    return y


def get_adjacency_matrix(G):
    """Returning adjacency matrix as node features."""
    A = nx.adjacency_matrix(G)
    A = A.todense()
    A = np.asarray(A)
    return A


def get_data_from_graph(G):
    """Getting data for pytorch from networkx graph"""
    data = from_networkx(G)
    return data


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    acc = float((out.argmax(-1) == data.y).sum() / data.y.shape[0])
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAT(100, 2)
    # model = torch.load("models/gat_100_2")
    model.load_state_dict(torch.load("models/gat_100_2_state_dict"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accuracy_graph = {}
    for i, data in zip(range(len(data_list)), data_list):
        data = data.to(device)
        accuracy_epoch = []
        for epoch in range(1, 200):
            loss = train(data)
            train_acc = test(data)
            if epoch % 100 == 0:
                accuracy_epoch.append(train_acc[0])
        accuracy_graph[i] = accuracy_epoch
    print(json.dumps(accuracy_graph, sort_keys=True, indent=4))
    # torch.save(model, "models/gat_100_2")
    torch.save(model.state_dict(), "models/gat_100_2_state_dict")

    # Print model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])
