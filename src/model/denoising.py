import torch
import numpy as np
from torch_geometric.nn import GINConv
from torch_geometric.nn import GCNConv

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, aggr

from torch_geometric.utils import add_self_loops
import torch.nn.functional as F

class SignDenoising2(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        torch.manual_seed(1234567)

        self.gin1 = GINConv(
            Sequential(Linear(input_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels)),
            train_eps=True)

        self.gin2 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, output_channels)),
            train_eps=True)

    def forward(self, x, edge_index):
        x = self.gin1(x, edge_index)
        x = self.gin2(x, edge_index)
        #x = self.gin2(x, edge_index)
        #x = self.gin2(x, edge_index)
        return x

class SignDenoising(torch.nn.Module):
    def __init__(self, hidden_channels, num_features):
        super().__init__()
        torch.manual_seed(1234567)
        self.global_pool = aggr.MeanAggregation()
        self.conv1 = GCNConv(num_features * 2, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, 2, add_self_loops=False)


    def forward(self, x, edge_index):
        neighbour_information = self.global_pool(x)
        complete_information = torch.cat([x, neighbour_information], dim=0)
        x = self.conv1(complete_information, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x