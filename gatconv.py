import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class Net(torch.nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.conv1 = GATConv(in_channels, 100, dropout=0.4)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(100, 32, dropout=0.4)
        self.conv3 = GATConv(32, out_channels, concat=False, dropout=0.4)


    def forward(self, x, data):

        x = F.elu(self.conv1(data.x, data.edge_index))
        # x = F.dropout(x, p=0.4, training=self.training)
        x = F.elu(self.conv2(x, data.edge_index))
        # x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv3(x, data.edge_index)

        return F.softmax(x, dim=-1)