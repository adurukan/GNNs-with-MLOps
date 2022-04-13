import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from helper import get_data, test_data, report_training_accuracy
from gat_net import GAT
"""
In this script, model will be selected and it will be evaluated with the desired data.
"""

with open("logger.txt", "w") as outfile:
    outfile.write("evaluate.py -> Imports are successful.")

data_list = get_data(test_data, "test_data")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("models/gat_300_2").to(device)
acc_graph = {}

def evaluate(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask'):
        acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
        accs.append(acc)
    return accs
    #         acc_graph[i] = acc_data
    # report_training_accuracy(acc_graph)

for data, i in zip(data_list, range(len(data_list))):
    accs = evaluate(data)
    acc_graph[i] = accs
print(acc_graph)
print(type(acc_graph))
#report_training_accuracy(acc_graph)
# acc_graph = evaluate()
# print(acc_graph)