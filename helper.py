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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

graph_data = [f for f in listdir("train_data") if isfile(join("train_data", f))]
test_data = [f for f in listdir("test_data") if isfile(join("test_data", f))]


def return_labels(G):
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
        G = nx.read_gpickle(f"train_data/{file_}")
        data = get_data_from_graph(G)
        A = get_adjacency_matrix(G)
        data.x = torch.from_numpy(A).float()
        data.y = return_labels(G)
        data.train_mask, data.val_mask, data.test_mask = retrieve_masks(data.y)
        # data.num_features = data.x.shape[1]
        data_list.append(data)
    train_dataset = data_list[len(data_list) // 10 :]
    test_dataset = data_list[: len(data_list) // 10]
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10)
    return train_loader, test_loader


def return_embeddings(G):
    embeddings = np.array(list(dict(G.degree()).values()))
    scale = StandardScaler()
    embeddings = scale.fit_transform(embeddings.reshape(-1, 1))
    embeddings = embeddings.reshape(1, -1)
    I = np.identity(embeddings[0].shape[0])
    for i in range(embeddings[0].shape[0]):
        I[i][i] = embeddings[0][i]
    return I


def get_dist_matrix(G):

    dist = nx.floyd_warshall_numpy(G)
    where_are_inf = np.isinf(dist)
    dist[where_are_inf] = dist[np.isfinite(dist)].max() * 100

    return dist


def get_degree_matrix(G):
    degrees = [val for (node, val) in G.degree()]
    I = np.identity(len(degrees))
    for i in range(len(degrees)):
        I[i][i] = degrees[i]
    return I


def get_features(G):
    A = get_adjacency_matrix(G)
    X = np.matrix([[i, -i] for i in range(A.shape[0])], dtype=float)
    I = np.matrix(np.eye(A.shape[0]))
    A_hat = A + I
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = np.matrix(np.diag(D_hat))
    features = D_hat**-1 * A_hat * X
    features = np.asarray(features)
    return features


def get_laplacian(G):
    G = nx.DiGraph(G)
    laplacian = nx.directed_laplacian_matrix(G)
    return laplacian


def get_incidence_matrix(G):
    incidences = nx.incidence_matrix(
        G, nodelist=list(G.nodes()), edgelist=list(G.edges()), oriented=True
    )
    return incidences


def tsvd(features):
    tsvd = TruncatedSVD(500)
    tsvd_features = tsvd.fit_transform(features)
    print(f"shape of tsvd_features: {tsvd_features.shape}")
    return tsvd_features


def get_data(graph_data, folder_path):
    data_list = []
    for file_ in graph_data:
        if "gpickle" in file_:
            G = nx.read_gpickle(f"{folder_path}/{file_}")
            data = get_data_from_graph(G)
            A = get_adjacency_matrix(G)
            # print(f"A: {A.shape} \t {type(A)}")
            embeddings = return_embeddings(G)
            # print(f"embeddings: {embeddings.shape} \t {type(embeddings)}")
            distance = get_dist_matrix(G)
            # print(f"distance: {distance.shape} \t {type(distance)}")
            degrees = get_degree_matrix(G)
            # print(f"degrees: {degrees.shape} \t {type(degrees)}")
            features = get_features(G)
            # print(f"features: {features.shape} \t {type(features)}")
            laplacian = get_laplacian(G)
            # print(f"laplacian: {laplacian.shape} \t {type(laplacian)}")
            incidences = get_incidence_matrix(G)
            # print(f"incidences: {incidences.shape} \t {type(incidences)}")
            incidences = tsvd(incidences)
            # print(f"incidences: {incidences.shape} \t {type(incidences)}")
            data.x = torch.from_numpy(incidences).float()
            data.y = return_labels(G)
            data.train_mask, data.val_mask, data.test_mask = retrieve_masks(data.y)
            data_list.append(data)
    return data_list


def report_training_accuracy(accuracy_dict):
    metrics = [f for f in listdir("metrics") if isfile(join("metrics", f))]
    if metrics == []:
        with open(f"metrics/0.json", "w") as outfile:
            json.dump(accuracy_dict, outfile, indent=8)
    else:
        new_file_name = (
            str(max(list(map(lambda s: int(s.split(".")[0]), metrics))) + 1) + ".json"
        )  # ['1.json', '3.json', '2.json'] -> '4.json'
        with open(f"metrics/{new_file_name}", "w") as outfile:
            json.dump(accuracy_dict, outfile, indent=8)


def filter_edge(nodelist, edge):
    return all(e in nodelist for e in [edge[0], edge[1]])


def create_plot(G, nodelist, colors, name):
    """Takes in a Graph, color string or list and name of file to save plot in"""
    pos = nx.random_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist, node_color=colors, node_size=100, alpha=1)
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
    )
    ax = plt.gca()
    for e in G.edges(nodelist):
        if filter_edge(nodelist, e):
            ax.annotate(
                "",
                xy=pos[e[0]],
                xycoords="data",
                xytext=pos[e[1]],
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="->",
                    color="0.5",
                    shrinkA=5,
                    shrinkB=5,
                    patchA=None,
                    patchB=None,
                    connectionstyle="arc3,rad=rrr".replace("rrr", str(0.3 * 0)),
                ),
            )
    plt.axis("off")
    plt.savefig("graphplots/real_diamonds/%s.png" % (name))
    plt.close()
