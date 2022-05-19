from networkx import karate_club_graph, to_numpy_matrix
import numpy as np
import torch
import torch.nn.functional as F
from helper import get_data, graph_data
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx

# zkc = karate_club_graph()
# order = sorted(list(zkc.nodes()))
# print(f"order: {order}")
data_list = get_data(graph_data, "train_data")
data = data_list[0]
print(data)
order = sorted(list(range(100)))
G = to_networkx(data)
# A = to_numpy_matrix(zkc, nodelist=order)
# I = np.eye(zkc.number_of_nodes())
A = nx.adjacency_matrix(G)
A = A.todense()
A = np.asarray(A)
print(data.num_nodes)
I = np.eye(data.num_nodes)
A_hat = A + I
print(f"A_hat: {A_hat.shape}")
D_hat = np.array(np.sum(A_hat, axis=0))[0]
print(f"D_hat: {D_hat.shape}")
D_hat = np.matrix(np.diag(D_hat))
print(f"D_hat: {D_hat.shape}")

W_1 = np.random.normal(loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
print(f"W_1: {W_1.shape}")
W_2 = np.random.normal(loc=0, size=(W_1.shape[1], 2))
print(f"W_2: {W_2.shape}")


def ReLU(x):
    return x * (x > 0)


def gcn_layer(A_hat, D_hat, X, W):

    sth = D_hat**-1 * A_hat * X * W
    sth = np.asarray(sth)
    print(type(sth))
    return ReLU(sth)


H_1 = gcn_layer(A_hat, D_hat, I, W_1)
print(f"H_1: {H_1.shape} and type {type(H_1)}")
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
output = H_2
print(f"output: {output.shape} and type {type(output)}")

feature_representations = {node: np.array(output)[node] for node in zkc.nodes()}
print(feature_representations)
