import networkx as nx
from sklearn import cluster
from helper import create_plot, get_adjacency_matrix
from scipy.sparse import csc_array
import numpy as np

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={"float_kind": float_formatter})
G = nx.DiGraph()
edges_list = [
    (2, 1),
    (3, 1),
    (2, 3),
    (4, 1),
    (2, 4),
    (3, 4),
    (5, 4),
    (5, 6),
    (5, 7),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (10, 12),
]
G.add_edges_from(edges_list)
create_plot(G, list(G.nodes()), "#00ffff", "clustering_cooefficient")

clustering_coefficient = nx.clustering(G)
clustered_coefficients = []
for i in sorted(clustering_coefficient.items()):
    clustered_coefficients.append(i[1])
print(clustered_coefficients)
incidences = nx.incidence_matrix(
    G, nodelist=list(G.nodes()), edgelist=list(G.edges()), oriented=True
)

A = get_adjacency_matrix(G)
I = np.identity(A.shape[0])
A = A + I
A_clustered = np.empty([A.shape[0], A.shape[1]])
for i in range(A.shape[0]):
    A_clustered[i] = A[i] + clustered_coefficients


clustered_matrix = np.zeros([A.shape[0], A.shape[1]])
for i in range(len(clustered_coefficients)):
    # print(f"k: {k}")
    # print(f"v: {v}")
    clustered_matrix[i, i] = clustered_coefficients[i]

# print(
#     f"clustered_coefficients with type: {type(clustered_coefficients)}: \n {clustered_coefficients}"
# )
# print(
#     f"incidences \n type: {type(incidences.toarray())}\n shape: {incidences.toarray().shape} \n {incidences.toarray()}"
# )
# print(f"Adjacency \n type: {type(A)} \n shape: {A.shape} \n {A}")

# print(
#     f"Adjacency_clustered \n type: {type(A_clustered)} \n shape: {A_clustered.shape} \n {A_clustered}"
# )

print(f"clustered_matrix with type: {type(clustered_matrix)}: \n {clustered_matrix}")
