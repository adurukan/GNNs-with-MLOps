import numpy as np
import networkx as nx
from helper import get_data_from_graph


G=nx.read_gpickle(r"C:\Users\DE124507\GNNs-with-MLOps\train_data\dataset_0_D.gpickle")

def get_dist_matrix(G):

    dist = nx.floyd_warshall_numpy(G)
    where_are_inf = np.isinf(dist)
    dist[where_are_inf] = dist[np.isfinite(dist)].max() * 100

    return dist


def see_full_dmatrix(dist):

    np.set_printoptions(threshold=np.inf)

    return print(dist)

