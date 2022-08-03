import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import EllipticBitcoinDataset
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
import matplotlib.pyplot as plt
import sys
from gcn_net import GCN
from gatconv import GATConv
warnings.filterwarnings("ignore")


# Load Dataframe
df_edge = pd.read_csv("elliptic/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
df_class = pd.read_csv("elliptic/elliptic_bitcoin_dataset/elliptic_txs_classes.csv")
df_features = pd.read_csv(
    "elliptic/elliptic_bitcoin_dataset/elliptic_txs_features.csv", header=None
)

# Setting Column name
df_features.columns = (
    ["id", "time step"]
    + [f"trans_feat_{i}" for i in range(93)]
    + [f"agg_feat_{i}" for i in range(72)]
)

# print("Number of edges: {}".format(len(df_edge)))
all_nodes = list(
    set(df_edge["txId1"])
    .union(set(df_edge["txId2"]))
    .union(set(df_class["txId"]))
    .union(set(df_features["id"]))
)
nodes_df = pd.DataFrame(all_nodes, columns=["id"]).reset_index()

# print("Number of nodes: {}".format(len(nodes_df)))

df_edge = (
    df_edge.join(
        nodes_df.rename(columns={"id": "txId1"}).set_index("txId1"),
        on="txId1",
        how="inner",
    )
    .join(
        nodes_df.rename(columns={"id": "txId2"}).set_index("txId2"),
        on="txId2",
        how="inner",
        rsuffix="2",
    )
    .drop(columns=["txId1", "txId2"])
    .rename(columns={"index": "txId1", "index2": "txId2"})
)
df_class = (
    df_class.join(
        nodes_df.rename(columns={"id": "txId"}).set_index("txId"),
        on="txId",
        how="inner",
    )
    .drop(columns=["txId"])
    .rename(columns={"index": "txId"})[["txId", "class"]]
)
df_features = (
    df_features.join(nodes_df.set_index("id"), on="id", how="inner")
    .drop(columns=["id"])
    .rename(columns={"index": "id"})
)
df_features = df_features[["id"] + list(df_features.drop(columns=["id"]).columns)]
# print(f"df_edge: \n {df_edge.head()}")
# print(f"df_class: \n {df_class.head()}")
# print(f"df_features: \n {df_features.head()}")

df_edge_time = df_edge.join(
    df_features[["id", "time step"]].rename(columns={"id": "txId1"}).set_index("txId1"),
    on="txId1",
    how="left",
    rsuffix="1",
).join(
    df_features[["id", "time step"]].rename(columns={"id": "txId2"}).set_index("txId2"),
    on="txId2",
    how="left",
    rsuffix="2",
)
df_edge_time["is_time_same"] = df_edge_time["time step"] == df_edge_time["time step2"]
df_edge_time_fin = df_edge_time[["txId1", "txId2", "time step"]].rename(
    columns={"txId1": "source", "txId2": "target", "time step": "time"}
)

df_features.drop(columns=["time step"]).to_csv(
    "elliptic/elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv",
    index=False,
    header=None,
)
df_class.rename(columns={"txId": "nid", "class": "label"})[
    ["nid", "label"]
].sort_values(by="nid").to_csv(
    "elliptic/elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv",
    index=False,
    header=None,
)
df_features[["id", "time step"]].rename(columns={"id": "nid", "time step": "time"})[
    ["nid", "time"]
].sort_values(by="nid").to_csv(
    "elliptic/elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv",
    index=False,
    header=None,
)
df_edge_time_fin[["source", "target", "time"]].to_csv(
    "elliptic/elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv",
    index=False,
    header=None,
)


node_label = (
    df_class.rename(columns={"txId": "nid", "class": "label"})[["nid", "label"]]
    .sort_values(by="nid")
    .merge(
        df_features[["id", "time step"]].rename(
            columns={"id": "nid", "time step": "time"}
        ),
        on="nid",
        how="left",
    )
)
node_label["label"] = (
    node_label["label"].apply(lambda x: "3" if x == "unknown" else x).astype(int) - 1
)
# print(f"node_label: \n {node_label.head()}")

merged_nodes_df = node_label.merge(
    df_features.rename(columns={"id": "nid", "time step": "time"}).drop(
        columns=["time"]
    ),
    on="nid",
    how="left",
)
# print(f"merged_nodes: \n {merged_nodes_df.head()}")

train_dataset = []
test_dataset = []
for i in range(49):
    nodes_df_tmp = merged_nodes_df[merged_nodes_df["time"] == i + 1].reset_index()
    nodes_df_tmp["index"] = nodes_df_tmp.index
    df_edge_tmp = (
        df_edge_time_fin.join(
            nodes_df_tmp.rename(columns={"nid": "source"})[
                ["source", "index"]
            ].set_index("source"),
            on="source",
            how="inner",
        )
        .join(
            nodes_df_tmp.rename(columns={"nid": "target"})[
                ["target", "index"]
            ].set_index("target"),
            on="target",
            how="inner",
            rsuffix="2",
        )
        .drop(columns=["source", "target"])
        .rename(columns={"index": "source", "index2": "target"})
    )
    x = torch.tensor(
        np.array(
            nodes_df_tmp.sort_values(by="index").drop(columns=["index", "nid", "label"])
        ),
        dtype=torch.float,
    )
    edge_index = torch.tensor(
        np.array(df_edge_tmp[["source", "target"]]).T, dtype=torch.long
    )
    edge_index = to_undirected(edge_index)
    mask = nodes_df_tmp["label"] != 2
    y = torch.tensor(np.array(nodes_df_tmp["label"]))

    if i + 1 < 35:
        data = Data(x=x, edge_index=edge_index, train_mask=mask, y=y)
        train_dataset.append(data)
    else:
        data = Data(x=x, edge_index=edge_index, test_mask=mask, y=y)
        test_dataset.append(data)




############################################ TRAIN ############################################
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# print(f"train_loader: \n {train_loader}")
# print(f"test_loader: \n {test_loader}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GATConv(in_channels=data.num_node_features, out_channels=2)
model.to(device)

patience = 50
lr = 0.001
weight_decay = 0
epoches = 750  #Default: 1000

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([9.245, 1]).to(device))

train_losses = []
val_losses = []
train_acc = []
val_acc = []
f1_illicit = []
precisions_illicit = []
precisions_licit = []
recalls_illicit = []
iterations = []

for epoch in range(epoches):
    model.train()
    train_loss = 0
    # predict_class = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)   #.reshape(data.y.size(dim=0)).float()
        train_acc_epoch = float((out.argmax(-1) == data.y).sum() / data.y.shape[0])
        loss = criterion(out[data.train_mask], data.y[data.train_mask].float().long())
        _, pred = out[data.train_mask].max(dim=1)
        # pred = out[data.train_mask]
        # predict_class = [1 if i > 0.5 else 0 for i in pred]
        # pred = torch.tensor(predict_class)
        loss.backward()
        train_loss += loss.item() * data.num_graphs
        optimizer.step()
        test_data = data
    train_loss /= len(train_loader.dataset)


    if (epoch + 1) % 1 == 0: #Default: 50
        model.eval()
        ys, preds = [], []
        val_loss = 0
        for data in test_loader:
            data = data.to(device)
            out = model(data)   #.reshape(data.y.size(dim=0)).float()
            val_acc_epoch = float((out.argmax(-1) == data.y).sum() / data.y.shape[0])
            loss = criterion(out[data.test_mask], data.y[data.test_mask].float().long())
            val_loss += loss.item() * data.num_graphs
            _, pred = out[data.test_mask].max(dim=1)
            # pred = out[data.test_mask]
            # predict_class = [1 if i > 0.5 else 0 for i in pred]
            # pred = torch.tensor(predict_class)
            ys.append(data.y[data.test_mask].cpu())
            preds.append(pred.cpu())

#Compute Losses & Metrics in current epoch
        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
        val_loss /= len(test_loader.dataset)

        if1 = f1_score(y, pred, average='binary', pos_label = 0)
        precision_illicit = precision_score(y, pred, average='binary', pos_label = 0)
        precision_licit = precision_score(y, pred, average='binary', pos_label = 1)
        recall_illicit = recall_score(y, pred, average='binary', pos_label = 0)
        recall_licit = recall_score(y, pred, average='binary', pos_label = 1)
        
#Append metrics to metrics of past epochs
        iterations.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        f1_illicit.append(if1)
        train_acc.append(train_acc_epoch)
        val_acc.append(val_acc_epoch)
        precisions_illicit.append(precision_illicit)
        precisions_licit.append(precision_licit)
        recalls_illicit.append(recall_illicit)

        if (epoch + 1) % 50 == 0: #Default: 50
            print(
                "Epoch: {:02d}, Train_Loss: {:.4f}, Val_Loss: {:.4f}, Train_Accuracy: {:.4f}, Val_Accuracy: {:.4f}, Precision_Illicit: {:.4f}, Recall_Illicit: {:.4f}, F1_Illicit: {:.4f}".format(
                    epoch + 1, train_loss, val_loss, train_acc_epoch, val_acc_epoch, precision_illicit, recall_illicit, precision_licit, if1
                )
            )

#Create DataFrames
loss_results = pd.DataFrame(columns = ['Iteration', 'Train_Loss', 'Val_Loss'])
acc_results = pd.DataFrame(columns = ['Iteration', 'Train_Accurancy', 'Val_Accurancy'])
metrics_results = pd.DataFrame(columns = ['Iteration', 'Precision_Illicit', 'Precision_Licit', 'Recall_Illicit', 'Illicit_F1'])

#Collect & Store Losses & accuracies
loss_results['Iteration'] = iterations
loss_results['Train_Loss'] = train_losses
loss_results['Val_Loss'] = val_losses

acc_results['Iteration'] = iterations
acc_results['Train_Accurancy'] = train_acc_epoch
acc_results['Val_Accurancy'] = val_acc_epoch

#Collect & Store Metrics in Metrics-Results-DataFrame
metrics_results['Iteration'] = iterations
metrics_results['Precision_Illicit'] = precisions_illicit
metrics_results['Precision_Licit'] = precisions_licit
metrics_results['Recall_Illicit'] = recalls_illicit
metrics_results['Illicit_F1'] = f1_illicit

#Plot Loss-Results-DataFrame Columns vs Iterations Column
acc_results.plot(x = 'Iteration', y = ['Train_Accurancy', 'Val_Accurancy'])
plt.savefig('acc_results.png')
loss_results.plot(x = 'Iteration', y = ['Train_Loss', 'Val_Loss'])
plt.savefig('loss_results.png')

#Plot Metrics-Results-DataFrame Columns vs Iterations Column 
metrics_results.plot(x = 'Iteration', y = ['Precision_Illicit', 'Precision_Licit', 'Recall_Illicit', 'Illicit_F1'])
plt.savefig('metrics_results.png')
    
