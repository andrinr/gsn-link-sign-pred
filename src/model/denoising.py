import torch
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F

class SignDenoising(torch.nn.Module):
    def __init__(self, hidden_channels, num_features):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)

    def forward(self, diffused_signs, embeddings, edge_index):
        

        return 0