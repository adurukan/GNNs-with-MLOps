import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, use_skip=False):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], 100)
        self.conv3 = GCNConv(100, 32)
        self.conv4 = GCNConv(32, 2)
        self.use_skip = use_skip
        if self.use_skip:
            self.weight = torch.nn.init.xavier_normal_(
                torch.nn.Parameter(torch.Tensor(num_node_features, 2))
            )

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.leaky_relu(x, 0.1)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = F.leaky_relu(x, 0.1)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, data.edge_index)
        x = F.leaky_relu(x, 0.1)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv4(x, data.edge_index)
        if self.use_skip:
            x = F.softmax(x + torch.matmul(data.x, self.weight), dim=-1)
        else:
            x = F.softmax(x, dim=-1)
        return x

    def embed(self, data):
        x = self.conv1(data.x, data.edge_index)
        return x
