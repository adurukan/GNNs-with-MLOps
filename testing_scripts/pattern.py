import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import GNNBenchmarkDataset

dataset = GNNBenchmarkDataset(root="/tmp/Pattern", name="PATTERN")
data = dataset[0]

print(f"data: \n {data}")

print(data.x[10:20])
print(data.y[10:20])
