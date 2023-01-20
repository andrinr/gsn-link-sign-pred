import torch
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, aggr

from torch_geometric.utils import add_self_loops
import torch.nn.functional as F

class SignDenoising(torch.nn.Module):
    def __init__(self, hidden_channels, num_features):
        super().__init__()
        torch.manual_seed(1234567)
        self.global_pool = aggr.SoftmaxAggregation(learn=True)
        self.conv1 = GCNConv(num_features * 2, hidden_channels, add_self_loops=False)
        self.conv2 = GCNConv(hidden_channels, 2, add_self_loops=False)


    def forward(self, x, edge_index):
        neighbour_information = self.global_pool(x)
        print(neighbour_information.shape, x.shape)
        complete_information = torch.cat([x, neighbour_information], dim=1)
        x = self.conv1(complete_information, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super().__init__(aggr='max')
        
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + 3, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))
        
    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.