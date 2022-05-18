import numpy as np
import torch
from helper import get_data, test_data, report_training_accuracy

# from gat_net import GAT
# from torch_geometric.nn import GATConv
"""
In this script, model will be selected and it will be evaluated with the desired data.
"""

with open("logger.txt", "w") as outfile:
    outfile.write("evaluate.py -> Imports are successful.")

data_list = get_data(test_data, "test_data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("models/gat_100_2").to(device)
acc_graph = {}


def evaluate(data):
    """Script to evaluate accuracies."""
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    real_one = data.y.numpy()
    predicted_one = out.argmax(-1).numpy()

    acc = float((out.argmax(-1) == data.y).sum() / data.y.shape[0])
    accs.append(acc)
    return real_one, predicted_one, accs


if __name__ == "__main__":
    real_ones = []
    predicted_ones = []
    for data, i in zip(data_list, range(len(data_list))):
        real_one, predicted_one, accs = evaluate(data)
        real_ones.append(real_one)
        predicted_ones.append(predicted_one)
        acc_graph[i] = accs

    report_training_accuracy(acc_graph)

    for i, j in zip(real_ones, predicted_ones):
        print(np.where(i == 1))
        print(np.where(j == 1))
