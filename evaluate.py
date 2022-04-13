import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from helper import get_data, test_data
from gat_net import GAT
"""
In this script, model will be selected and it will be evaluated with the desired data.
"""

with open("logger.txt", "w") as outfile:
    outfile.write("evaluate.py -> Imports are successful.")

data_list = get_data(test_data, "test_data")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("models/gat_300_2").to(device)

def evaluate(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask'):
        acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
    return acc
    #         acc_graph[i] = acc_data
    # report_training_accuracy(acc_graph)

for data, i in zip(data_list, range(len(data_list))):
    acc = evaluate(data)
    print(acc)
# acc_graph = evaluate()
# print(acc_graph)