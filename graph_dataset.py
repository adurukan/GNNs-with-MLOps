import torch

from torch_geometric.data import Data, Dataset, InMemoryDataset


class MyTestDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(None)
        self.data, self.slices = self.collate(data_list)
