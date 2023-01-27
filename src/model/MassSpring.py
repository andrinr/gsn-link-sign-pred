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
        
    def forward(self, position, edge_index, relaxed_lengths):
        return self.propagate(edge_index, position=position, undeformed=relaxed_lengths)

    def message(self, position_i, position_j, undeformed):
        spring = position_j - position_i
        deformed = torch.norm(spring, dim=1, keepdim=False)
        force = (deformed - undeformed) * self.stiffness *  torch.div(spring.T, deformed + 0.001)
        return force.T

    def __repr__(self) -> str:
        return super().__repr__() + f'({self.stiffness})'