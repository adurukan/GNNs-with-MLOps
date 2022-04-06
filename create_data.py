import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
import matplotlib.pyplot as plt
from random import randint
import os
import json

"""
In this script, data of various nodes, numbers of diamond patterns, and probability are created and all those are to be stored in data/
"""
with open("logger.txt", "w") as outfile:
    outfile.write("crea_data.py -> Imports are successful.")