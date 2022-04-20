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
model = torch.load("models/gat_300_2").to(device)
acc_graph = {}


def evaluate(data):
    """Script to evaluate accuracies."""
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    acc = float((out.argmax(-1) == data.y).sum() / data.y.shape[0])
    accs.append(acc)
    return accs


if __name__ == "__main__":
    for data, i in zip(data_list, range(len(data_list))):
        accs = evaluate(data)
        acc_graph[i] = accs

    report_training_accuracy(acc_graph)
