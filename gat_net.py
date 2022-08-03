import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import sys


class GAT(torch.nn.Module):
    def __init__(self, out_channels):

        super().__init__()
        self.conv1 = GATConv(-1, 256, heads=10, dropout=0.2)
        self.conv2 = GATConv(256 * 10, 56, heads=8, dropout=0.2)
        self.conv3 = GATConv(56 * 8, 32, heads=4, dropout=0.2)
        self.conv4 = GATConv(32 * 4, out_channels, heads=1, concat=False)


    def forward(self, x, edge_index):

        x = F.leaky_relu(self.conv1(x, edge_index), 0.1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.conv2(x, edge_index), 0.1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.conv3(x, edge_index), 0.1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.conv4(x, edge_index), 0.1)
        # x = self.conv5(x, edge_index)
        return F.softmax(x, dim=1)


def create_save_model():
    in_channels = int(sys.argv[1])
    out_channels = int(sys.argv[2])
    model = GAT(in_channels=in_channels, out_channels=out_channels)
    torch.save(model, f"models/gat_{in_channels}_{out_channels}")


# create_save_model()
