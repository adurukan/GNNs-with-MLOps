import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import sys


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 16)
        self.conv3 = GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=-1)


# def create_save_model():
#     in_channels = int(sys.argv[1])
#     out_channels = int(sys.argv[2])
#     model = GAT(in_channels=in_channels, out_channels=out_channels)
#     torch.save(model, f"models/gat_{in_channels}_{out_channels}")


# create_save_model()
