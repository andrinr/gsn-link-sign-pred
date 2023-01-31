import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool
from model import OpinionLayer
from torch_geometric.nn import GCNConv

class SpringLoss(torch.nn.Module):
    def __init__(self, edge_index, relaxed_lengths, sign, stiffness):
        super().__init__()
        self.edge_index = edge_index
        self.relaxed_lengths = relaxed_lengths
        self.sign = sign
        self.stiffness = stiffness
        self.adjaceny_map 
    
    def forward(self, output, target):
        for i in range(output.shape[0]):
            position = output[i]
            position_i = position[self.edge_index[0]]
            position_j = position[self.edge_index[1]]
            spring = position_j - position_i
            length = torch.norm(spring, dim=1, keepdim=False)
            normalized = torch.div(spring.T, length + 0.001)
            attraction = (length - self.relaxed_lengths) * self.stiffness * normalized
            retraction = -torch.nn.functional.relu(self.relaxed_lengths - length, inplace=True) * self.stiffness * normalized
            force = torch.where(self.sign == 1, attraction, retraction)
            return torch.mean(torch.abs(output - target))
        spring = position_j - position_i
        length = torch.norm(spring, dim=1, keepdim=False)
        normalized = torch.div(spring.T, length + 0.001)
        attraction = (length - relaxed_lengths) * self.stiffness * normalized
        retraction = -torch.nn.functional.relu(relaxed_lengths - length, inplace=True) * self.stiffness * normalized
        force = torch.where(sign == 1, attraction, retraction)
        return torch.mean(torch.abs(output - target))

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

        
