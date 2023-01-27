import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool
from model import OpinionLayer
from torch_geometric.nn import GCNConv

class Opinion(torch.nn.Modue):
    def __init__(
        self,
        pe_channels : int,
        hidden_channels : int,
    ) -> None:
        super().__init__()

        # Propagation
        self.conv1 = OpinionLayer(pe_channels, hidden_channels)
        self.conv2 = GCNConv((hidden_channels + pe_channels) * 2, 2)

    def forward(self, pos, hidden, sign, edge_index):
        
        x = self.conv1(x, edge_index, edge_index)


        return x

        
