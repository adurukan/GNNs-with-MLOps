import torch
import torch.nn.functional as F
import sklearn as sk
from torch_geometric.loader import DataLoader
from graph_dataset import MyTestDataset
from gat_net import GAT
from gcn_net import GCN
from gatconv import GATConv
import pandas as pd
from helper import get_data, graph_data, test_data
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

"""
In this script, model&data will be loaded and model will be trained accordingly.
"""
with open("logger.txt", "w") as outfile:
    outfile.write("train.py -> Imports are successful.")

def train():
    total_loss = total_examples = total_acc = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y_pred = model(data.x, data.edge_index)      #.reshape(data.y.size(dim=0)).float()
        # print(f'prediciton shape is {prediction_.shape}')
        # print(f'Target has {data.y.sum()} suspicious nodes with shape {data.y.shape}')
        acc = float((y_pred.argmax(-1) == data.y).sum() / data.y.shape[0])
        loss = F.cross_entropy(y_pred, data.y.float().long(), torch.tensor([0.2, 0.8]))
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_acc += float(acc) * data.num_graphs
        total_examples += data.num_graphs
        train_loss = total_loss / total_examples
        train_acc = total_acc / total_examples
    return train_loss , train_acc


@torch.no_grad()         
def test(loader):
    # mse = []
    model.eval()
    cel = []
    accs_ = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        accs_.append(float((out.argmax(-1) == data.y).sum() / data.y.shape[0]))
        # out = out.reshape(data.y.size(dim=0)).float()
        # print(f'shape of out {out.shape} with type {type(out)}')
        # print(f'shape of target {data.y.shape} with type {type(data.y)}')
        cel.append(F.cross_entropy(out, data.y.long(), torch.tensor([0.2, 0.8])))
        val_loss, val_acc = float(sum(cel) / len(cel)), float(sum(accs_) / len(accs_))
    return val_loss, val_acc, out.argmax(-1), data.y


def compute_matrics(y_true, y_pred):
    
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return f1, precision, recall


def collect_all_metrics(train_acc, val_acc, train_loss, val_loss, f1, precision, recall, epoch):

    iterations.append(epoch + 1)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)    
    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    return iterations, train_accs, val_accs, train_losses, val_losses, f1s, precisions, recalls


def plot(iterations, train_losses, train_accs,  val_losses, val_accs, f1s, precisions, recalls):

    matplotlib.use('Agg')
    loss_results = pd.DataFrame(columns = ['Iteration', 'Train Loss', 'Validation Loss'])
    acc_results = pd.DataFrame(columns = ['Iteration', 'Train Accurancy', 'Validation Accuracy'])
    metrics_results = pd.DataFrame(columns = ['Iteration', 'Precision', 'Recall', 'F1'])

    loss_results['Iteration'] = iterations
    loss_results['Train Loss'] = train_losses
    loss_results['Validation Loss'] = val_losses

    acc_results['Iteration'] = iterations
    acc_results['Train Accuracy'] = train_accs
    acc_results['Validation Accuracy'] = val_accs

    metrics_results['Iteration'] = iterations
    metrics_results['Precision'] = precisions
    metrics_results['Recall'] = recalls
    metrics_results['F1'] = f1s

    loss_results.plot(x = 'Iteration', y = ['Train Loss', 'Validation Loss'])
    plt.savefig('loss_results.png')
    acc_results.plot(x = 'Iteration', y = ['Train Accuracy', 'Validation Accuracy'])
    plt.savefig('acc_results.png')
    metrics_results.plot(x = 'Iteration', y = ['Precision', 'Recall', 'F1'])
    plt.savefig('metrics_results.png')




if __name__ == "__main__":
    data_list_training = get_data(graph_data, "train_data")
    data_list_testing = get_data(test_data, "test_data")
    data_list = data_list_training + data_list_testing
    dataset = MyTestDataset(data_list)
    N = len(dataset)
    train_dataset = dataset[: int(N * 0.8)]
    val_dataset = dataset[int(N * 0.8) :]
    # test_dataset = dataset[int(N * 0.9) :]

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    f1s = []
    precisions = []
    recalls = []
    iterations = []
    for epoch in range(1, 701):
        train_loss, train_acc = train()
        val_loss, val_acc, y_pred, y_true = test(val_loader)
        # test_loss, test_acc = test(test_loader)
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch: {epoch:03d} \n"
                f"Training_loss: {train_loss:.4f} Training_accuracy: {train_acc: .4f} "
                f"Val_loss: {val_loss:.4f} Val_acc: {val_acc: .4f} "
                #f"Test_loss: {test_loss:.4f} Test_acc: {test_acc: .4f} "
            )
        f1, precision, recall = compute_matrics(y_true, y_pred)
        iterations, train_accs, val_accs, train_losses, val_losses, f1s, precisions, recalls = collect_all_metrics(train_acc, val_acc, train_loss, val_loss, f1, precision, recall, epoch)

    plot(iterations, train_losses, train_accs,  val_losses, val_accs, f1s, precisions, recalls)