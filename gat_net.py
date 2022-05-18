import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import sys


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GATConv(in_channels, 10, heads=10, dropout=0.5)
        self.conv2 = GATConv(10 * 10, 10, heads=8, dropout=0.5)
        self.conv3 = GATConv(10 * 8, 10, heads=4, dropout=0.5)
        self.conv4 = GATConv(10 * 4, 10, heads=1, dropout=0.5)
        self.conv5 = GATConv(10 * 1, out_channels, heads=1, concat=False, dropout=0.5)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv4(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv5(x, edge_index)
        return F.log_softmax(x, dim=-1)


def create_save_model():
    in_channels = int(sys.argv[1])
    out_channels = int(sys.argv[2])
    model = GAT(in_channels=in_channels, out_channels=out_channels)
    torch.save(model, f"models/gat_{in_channels}_{out_channels}")


# create_save_model()
