import numpy as np
import torch
from helper import get_data, test_data, report_training_accuracy
from gat_net import GAT
from gcn_net import GCN
import warnings
import os
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
"""
In this script, model will be selected and it will be evaluated with the desired data.
"""

with open("logger.txt", "w") as outfile:
    outfile.write("evaluate.py -> Imports are successful.")

data_list = get_data(test_data, "test_data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path_state_dict = "models/gat_100_2_state_dict"
# model = GAT(100, 2)
# path_state_dict = "models/gcn_100_2_state_dict"
model = GCN(2)

# if os.path.isfile(path_state_dict):
#     print("State Dict exists")
#     model.load_state_dict(torch.load(path_state_dict))
# else:
#     pass
# print(f"Model: \n {model}")

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


def plot_eval(recalls, specifities, precisions, f_scores):
        matplotlib.use('Agg')
        metrics_results = pd.DataFrame(columns = ['Iteration', 'Recall', 'Specifity', 'Precision', 'f_score'])

        metrics_results['Iteration'] = iterations
        metrics_results['Recall'] = recalls
        metrics_results['Specifity'] = specifities
        metrics_results['Precision'] = precisions
        metrics_results['f_score'] = f_scores

        metrics_results.plot()
        metrics_results.plot(x = 'Iteration', y = ['Recall', 'Specifity', 'Precision', 'f_score'])
        plt.savefig('metrics_results.png')


if __name__ == "__main__":
    
    real_ones = []
    predicted_ones = []
    epoch = 0
    for data, i in zip(data_list, range(len(data_list))):
        real_one, predicted_one, accs = evaluate(data)
        real_ones.append(real_one)
        predicted_ones.append(predicted_one)
        # acc_graph[i] = accs
    
    for i, j in zip(real_ones, predicted_ones):
        print(np.where(i == 1))
        print(np.where(j == 1))

    iterations = []
    recalls = []
    specifities = []
    precisions = []
    f_scores = []

    for i, j in zip(real_ones, predicted_ones):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        real_pos = np.where(i == 1)
        real_neg = np.where(i == 0)
        predicted_pos = np.where(j == 1)
        predicted_neg = np.where(j == 0)
        for k in predicted_pos[0]:
            if k in real_pos[0]:
                TP = TP + 1
            else:
                FP = FP + 1
        for k in predicted_neg[0]:
            if k in real_neg[0]:
                TN = TN + 1
            else:
                FN = FN + 1

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        specifity = TN / (TN + FP)
        if TP + FN > 0:
            recall = TP / (TP + FN)
        else:
            recall = "No true positives and no false negatives"
        if TP + FP > 0:
            precision = TP / (TP + FP)
        else:
            precision = "No positives"
        if type(precision) == float and type(recall) == float:
            f_score = (2 * precision * recall) / (precision + recall) + 0.0001
        else:
            f_score = "Precision + Recall !> 0"

        recalls.append(recall)
        specifities.append(specifity)
        precisions.append(precision)
        f_scores.append(f_score)
        iterations.append(epoch + 1)


    try:
        acc_graph["recall"] = sum(recalls) / len(recalls)
    except:
        acc_graph["recall"] = "Not valid"
    try:
        acc_graph["specifity"] = sum(specifities) / len(specifities)
    except:
        acc_graph["specifity"] = "Not valid"
    try:
        acc_graph["precision"] = precisions
    except:
        acc_graph["precision"] = "Not valid"
    try:
        acc_graph["f_score"] = f_scores
    except:
        acc_graph["f_score"] = "Not valid"
    report_training_accuracy(acc_graph)

    plot_eval(recalls, specifities, precisions, f_scores)