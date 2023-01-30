import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential, Linear, ReLU

class MassSpring(MessagePassing):
    def __init__(self, 
        stiffness : float):
        super().__init__(aggr='add') 
        self.stiffness = stiffness
        
    def forward(self, position, edge_index, relaxed_lengths, sign):
        return self.propagate(edge_index, position=position, relaxed_lengths=relaxed_lengths, sign=sign)

    def message(self, position_i, position_j, relaxed_lengths, sign):
        spring = position_j - position_i
        length = torch.norm(spring, dim=1, keepdim=False)
        normalized = torch.div(spring.T, length + 0.001)
        attraction = (length - relaxed_lengths) * self.stiffness * normalized
        retraction = -torch.nn.functional.relu(relaxed_lengths - length, inplace=True) * self.stiffness * normalized
        force = torch.where(sign == 1, attraction, retraction)
        return force.T

    def __repr__(self) -> str:
        return super().__repr__() + f'({self.stiffness})'