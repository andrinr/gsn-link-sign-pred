import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential, Linear, ReLU

class Attraction(MessagePassing):
    def __init__(self, 
        stiffness : float):
        super().__init__(aggr='add') 
        self.stiffness = stiffness
    
     
    def forward(self, position, edge_index, sign):
        return self.propagate(edge_index, position=position, sign=sign)

    def message(self, position_i, position_j, sign):
        vector = position_j - position_i
        distance = torch.norm(vector, dim=1, keepdim=False)
        force = torch.div(vector.T, distance + 0.001) * ( 1 - 0.01 * distance) * sign
        return force.T

    def __repr__(self) -> str:
        return super().__repr__() + f'({self.stiffness})'