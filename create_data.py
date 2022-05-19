import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import from_networkx
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
import pickle
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
from random import randint
from os import listdir
from os.path import isfile, join
import json

# from visualize import create_plot

"""
In this script, data of various nodes, numbers of diamond patterns, and probability are created and all those are to be stored in data/
"""
with open("logger.txt", "w") as outfile:
    outfile.write("crea_data.py -> Imports are successful.")


def erdosrenyi_generator(n, p):
    """
    Args:
    n: given node number
    p: possibility for edge creation;
       note here p can be calculated given desired average connecting degree acd
       through math: p = acd / (n - 1), for n > 100 simplified as: p = acd / n

    """
    G_er = erdos_renyi_graph(n, p, directed=True)
    return G_er


def addedges(G, k):
    """generate multi-directed graph on basis of the passed-in (directed) gragh

    For each time, copy existing edges once if true; nothing happens while false
    (true/false generated by randint). We aim here 1.to randomly increase some of the existing edges;
    2.increase it to random number.
    We do this aiming at creating the raw training set background with sufficient randomness.

    Args:
    G: dealt network
    k: number of repeat time; choose 3 if there's no preference

    Returns:
    Multi-directed graph G with added edges
    """
    G = nx.MultiDiGraph(G)
    edgelistG = list(G.edges(data=False))

    for i in range(k):
        idx = [randint(0, 1) for p in range(0, len(edgelistG))]
        added_edge = []
        for j in range(len(edgelistG)):
            if idx[j]:
                added_edge.append(edgelistG[j])
        G.add_edges_from(added_edge)
    nx.set_node_attributes(G, 0, "suspicious")
    return G


# def visualize(G):
#     print("Generated background has" + str(len(G.edges)) + "edges.")
#     print(G.edges(data=False))
#     print(G.nodes)
#     nx.draw(G, with_labels=True, font_weight="bold")

#     plt.show()


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


def generateRandomLengthPath(BG, startNode, endNode, maxDepth, useBackground):
    nodes = list(BG.nodes)
    n = len(nodes)

    existingDiamonds = nx.get_node_attributes(BG, "suspicious")
    diamondNodes = [k for k, v in existingDiamonds.items() if v == 1]
    # print(f"diamondNodes: {diamondNodes}")

    v = []
    e = []
    # for i in range(randint(1, maxDepth)):
    for i in range(maxDepth):
        newNode = randint(0, n - 1)
        while newNode in diamondNodes:
            newNode = randint(0, n - 1)
        if not useBackground:
            while newNode in nodes:
                newNode = randint(n + 1, n + n + n)

        v.append(newNode)
    # print(f"v: {v}")
    for y in range(len(v) - 1):
        e.append((v[y], v[y + 1]))
    G = nx.MultiDiGraph(e)
    # print(f"start_node: {startNode}")
    # print(f"endNode: {endNode}")
    G.add_nodes_from([startNode, endNode])
    G.add_edges_from(
        [(v[0], startNode), (endNode, v[0]), (v[1], startNode), (endNode, v[1])]
    )
    # G.add_edges_from([(v[0], startNode), (endNode, v[-1])])
    return G


def generateDiamond(BG, startNode, endNode, splitDegree, maxDepth, useBackground):
    G = nx.MultiDiGraph()
    G.add_node(startNode)
    G.add_node(endNode)

    for i in range(splitDegree):
        pathGraph = generateRandomLengthPath(
            BG, startNode, endNode, maxDepth, useBackground
        )
        G.update(pathGraph)
    nx.set_node_attributes(G, 1, "suspicious")
    return G


def addPattern(G, pattern):
    G.update(pattern)


def return_labels(G):
    nodes = G.nodes
    labels = nx.get_node_attributes(G, "suspicious")
    y = []
    for label in labels.values():
        y.append(label)
    y = np.asarray(y)
    return y


if __name__ == "__main__":
    # initializing various variables: number of graphs to be generated, current graph number, max depth of paths within diamond
    # useBackground: boolean variable to decide whether the diamonds should consist of completely new nodes (False) or also use nodes already in the background (True)
    num_graphs = 50
    graph_number = 0
    useBackground = True
    addDiamonds = True
    maxPathDepth = 2

    for i in range(num_graphs):
        # diamond lst
        diamonds = []
        # reset Graph
        G = nx.MultiDiGraph
        # create an empty graph to hold diamonds
        dd = nx.MultiDiGraph()
        # generate new random number for number of diamonds and nodes in total
        num_diamonds = randint(6, 10)
        num_nodes = 100
        # Create Background Graph
        G_er = erdosrenyi_generator(n=num_nodes, p=3 / num_nodes)
        G = addedges(G_er, k=3)

        if addDiamonds:
            # run loop as many times as the number of Diamonds to be generated
            for d in range(num_diamonds):
                # generate new random numbers for split degree and start and end nodes
                splitDegree = 1
                startNode = randint(0, num_nodes // 2)
                endNode = randint(0, num_nodes - 1)
                # incase start and end happen to be the same
                while endNode == startNode:
                    endNode = randint(0, num_nodes - 1)
                # generate the diamond and add it to the graph
                diamond = generateDiamond(
                    G, startNode, endNode, splitDegree, maxPathDepth, useBackground
                )
                diamonds.extend(diamond.nodes)
                dd = nx.compose(dd, diamond)
                addPattern(G, diamond)
        with open(f"graphplots/real_diamonds/{i}.json", "w") as outfile:
            json.dump(diamonds, outfile, indent=8)
        create_plot(dd, list(diamonds), "#00ffff", "created-diamonds_" + str(i))
        # save graph in pickle file
        if graph_number <= int(num_graphs * 0.85):
            nx.write_gpickle(G, "train_data/dataset_%s_D.gpickle" % (graph_number))
        else:
            nx.write_gpickle(G, "test_data/dataset_%s_D.gpickle" % (graph_number))
        graph_number += 1
        # print(graph_number)
# print("graphs completed: " + str(graph_number))
