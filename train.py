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

"""
In this script, model&data will be loaded and model will be trained accordingly.
"""
with open("logger.txt", "w") as outfile:
    outfile.write("train.py -> Imports are successful.")