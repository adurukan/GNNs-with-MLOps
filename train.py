import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx

from torch_geometric.loader import DataLoader
from graph_dataset import MyTestDataset
from gat_net import GAT
from gcn_net import GCN
import networkx as nx
from helper import get_data, graph_data, test_data
import os
import math
import warnings

warnings.filterwarnings("ignore")

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


# @torch.no_grad()
# def test(data):
#     # model.eval()
#     out, accs = model(data.x, data.edge_index), []
#     acc = float((out.argmax(-1) == data.y).sum() / data.y.shape[0])
#     # accs.append(acc)
#     return acc


# def train(data):
#     model.train()
#     # optimizer.zero_grad()
#     out = model(data.x, data.edge_index)
#     loss = F.nll_loss(out, data.y)
#     # loss.backward()
#     # optimizer.step()
#     return loss


if __name__ == "__main__":
    data_list_training = get_data(graph_data, "train_data")
    data_list_testing = get_data(test_data, "test_data")
    data_list = data_list_training + data_list_testing
    dataset = MyTestDataset(data_list)
    N = len(dataset)
    train_dataset = dataset[: int(N * 0.8)]
    val_dataset = dataset[int(N * 0.8) : int(N * 0.9)]
    test_dataset = dataset[int(N * 0.9) :]

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GCN(500, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train():
        total_loss = total_examples = total_acc = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            # loss = F.mse_loss(out, data.y)
            acc = float((out.argmax(-1) == data.y).sum() / data.y.shape[0])
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
            total_acc += float(acc) * data.num_graphs
            total_examples += data.num_graphs
        return total_loss / total_examples, total_acc / total_examples
        # return math.sqrt(total_loss / total_examples)

    @torch.no_grad()
    def test(loader):
        # mse = []
        nll = []
        accs_ = []
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            accs_.append(float((out.argmax(-1) == data.y).sum() / data.y.shape[0]))
            # mse.append(F.mse_loss(out, data.y, reduction="none").cpu())
            nll.append(F.nll_loss(out, data.y).cpu())
        return float(sum(nll) / len(nll)), float(sum(accs_) / len(accs_))
        # return float(torch.cat(nll, dim=0).mean())
        # return float(torch.cat(mse, dim=0).mean().sqrt())

    for epoch in range(1, 201):
        train_loss, train_acc = train()
        val_loss, val_acc = test(val_loader)
        test_loss, test_acc = test(test_loader)
        print(
            f"Epoch: {epoch:03d} \n"
            f"Training_loss: {train_loss:.4f} Training_accuracy: {train_acc: .4f} "
            f"Val_loss: {val_loss:.4f} Val_acc: {val_acc: .4f} "
            f"Test_loss: {test_loss:.4f} Test_acc: {test_acc: .4f} "
        )
    # # path_state_dict = "models/gat_100_2_state_dict"
    # # model = GAT(100, 2)
    # path_state_dict = "models/gcn_100_2_state_dict"
    # model = GCN(500, 2)
    # # model = torch.load("models/gat_100_2")
    # if os.path.isfile(path_state_dict):
    #     print("State Dict exists")
    #     model.load_state_dict(torch.load(path_state_dict))
    # else:
    #     pass
    # print(f"Model: \n {model}")
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # # accuracy_graph = {}

    # for epoch in range(0, 10):
    #     optimizer.zero_grad()
    #     # print(f"Number of data: {i}")
    #     # print("------------------------")
    #     # accuracy_epoch = []
    #     acc_ = 0
    #     count = 0
    #     for i, data in zip(range(len(data_list)), data_list):
    #         data = data.to(device)
    #         loss = train(data)
    #         acc_ = acc_ + test(data)
    #         count = count + 1
    #     loss.backward()
    #     optimizer.step()
    #     acc_ = acc_ / count
    #     print(f"EPOCH: : {epoch}")
    #     print(f"accuracy: {acc_} \n")
    #     # if epoch % 100 == 0:
    #     #     print(f"EPOCH: {epoch}")
    #     #     print(f"accuracy: {train_acc[0]} \n")
    #     # accuracy_epoch.append(train_acc[0])
    # #    accuracy_graph[i] = accuracy_epoch
    # # print(json.dumps(accuracy_graph, sort_keys=True, indent=4))
    # # torch.save(model, "models/gat_100_2")
    # model.eval()
    # torch.save(model.state_dict(), path_state_dict)
