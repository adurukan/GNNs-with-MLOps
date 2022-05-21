import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, GNNBenchmarkDataset

# dataset = Planetoid(root="/tmp/Cora", name="Cora")
# data = dataset[0]
# print(f"num_features: {dataset.num_features} num_classes: {dataset.num_classes}")
# print(f"data: \n {data}")


# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         self.conv1 = GCNConv(in_channels, 100)
#         self.conv2 = GCNConv(100, 32)
#         self.conv3 = GCNConv(32, out_channels)

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.2, training=self.training)
#         x = F.relu(self.conv2(x, edge_index))
#         x = F.dropout(x, p=0.2, training=self.training)
#         x = self.conv3(x, edge_index)
#         return F.log_softmax(x, dim=-1)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model, data = GCN(dataset.num_features, dataset.num_classes).to(device), data.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)


# def train():
#     model.train()
#     optimizer.zero_grad()
#     F.nll_loss(
#         model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask]
#     ).backward()
#     optimizer.step()


# @torch.no_grad()
# def test():
#     model.eval()
#     log_probs, accs = model(data.x, data.edge_index), []
#     for _, mask in data("train_mask", "test_mask"):
#         pred = log_probs[mask].max(1)[1]
#         acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
#         accs.append(acc)
#     return accs


# for epoch in range(1, 201):
#     train()
#     train_acc, test_acc = test()
#     print(f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}")
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, 2)
        self.conv3 = GCNConv(2, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=-1)


import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # data_list_training = get_data(graph_data, "train_data")
    # data_list_testing = get_data(test_data, "test_data")
    # data_list = data_list_training + data_list_testing
    # dataset = MyTestDataset(data_list)
    # dataset = Planetoid(root="/tmp/Cora", name="Cora")
    dataset = GNNBenchmarkDataset(root="/tmp/Pattern", name="PATTERN")
    data = dataset[0]
    print(f"num_features: {dataset.num_features} num_classes: {dataset.num_classes}")
    print(f"data: \n {data}")
    N = len(dataset)
    print(f"N: {N}")
    train_dataset = dataset[: int(N * 0.7)]
    val_dataset = dataset[int(N * 0.7) : int(N * 0.8)]
    test_dataset = dataset[int(N * 0.8) :]
    print(f"train_dataset: \n {train_dataset}")
    print(f"val_dataset: \n {val_dataset}")
    print(f"test_dataset: \n {test_dataset}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GCN(dataset.num_features, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train():
        model.train()
        total_loss = total_examples = total_acc = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            acc = float((out.argmax(-1) == data.y).sum() / data.y.shape[0])
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
            total_acc += float(acc) * data.num_graphs
            total_examples += data.num_graphs
        return total_loss / total_examples, total_acc / total_examples

    @torch.no_grad()
    def test(loader):
        model.eval()
        nll = []
        accs_ = []
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            accs_.append(float((out.argmax(-1) == data.y).sum() / data.y.shape[0]))
            nll.append(F.nll_loss(out, data.y).cpu())
        return float(sum(nll) / len(nll)), float(sum(accs_) / len(accs_))

    for epoch in range(1, 201):
        train_loss, train_acc = train()
        val_loss, val_acc = test(val_loader)
        test_loss, test_acc = test(test_loader)
        print(
            f"Epoch: {epoch:03d} \n"
            f"Training_loss: {train_loss:.4f} Training_accuracy: {train_acc: .4f} "
            f"Val_loss: {val_loss:.4f} Val_acc: {val_acc: .4f} "
            f"Test_loss: {test_loss:.4f} Test_acc: {test_acc: .4f} "
        )
