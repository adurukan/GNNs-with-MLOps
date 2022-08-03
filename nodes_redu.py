import numpy as np
import networkx as nx
from helper import get_data_from_graph


G=nx.read_gpickle(r"C:\Users\DE124507\GNNs-with-MLOps\train_data\dataset_0_D.gpickle")
#print(G)
data = get_data_from_graph(G)
adj = nx.floyd_warshall_numpy(G)

where_are_inf = np.isinf(adj)
adj[where_are_inf] = adj[np.isfinite(adj)].max() * 100

np.set_printoptions(threshold=np.inf)
print(adj)

#suspicious = nx.get_node_attributes(G, 'suspicious')
#a = list(suspicious.values())
#print(a)
#suspicious_nodes = list(suspicious.keys()) [list (suspicious.values()).index (1)]
#print(suspicious_nodes)

#def distance_to_diamond(adj):

