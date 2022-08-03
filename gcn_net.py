import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import sys


class GCN(torch.nn.Module):
    def __init__(self, out_channels):

        super().__init__()
        self.conv1 = GCNConv(-1, 256)
        self.conv2 = GCNConv(256, 128)
        self.conv3 = GCNConv(128, 56)
        self.conv4 = GCNConv(56, 32)
        self.linear = Linear(32, out_channels)

    def forward(self, x, edge_index):

        x = F.leaky_relu(self.conv1(x, edge_index), 0.1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.conv2(x, edge_index), 0.1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.conv3(x, edge_index), 0.1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.conv4(x, edge_index), 0.1)
        
        return F.softmax(self.linear(x))

