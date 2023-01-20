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
            Sequential(Linear(input_channels, hidden_channels), ReLU(), Linear(hidden_channels, output_channels)),
            train_eps=True)

        self.gin2 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, output_channels)),
            train_eps=True)

    def forward(self, x, edge_index):
        x = self.gin1(x, edge_index)
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

class MP(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j