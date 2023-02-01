import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch.nn.functional import relu
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential, Linear, ReLU

class MassSpring(MessagePassing):
    def __init__(self, 
        stiffness : float,
        far : float,
        medium : float,
        close : float):
        super().__init__(aggr='add') 
        self.stiffness = stiffness
        self.far = far
        self.medium = 6.0
        self.close = close
        print(f"MassSpring: stiffness={stiffness}, far={far}, medium={medium}, close={close}")
        
    def forward(self, position, edge_index, sign):
        return self.propagate(edge_index, position=position, sign=sign)

    def message(self, position_i, position_j, sign):
        spring = position_j - position_i
        length = torch.norm(spring, dim=1, keepdim=False)
        normalized = torch.div(spring.T, length + 0.001)
        attraction = relu(length - self.close) * self.stiffness * normalized
        regular = (length - self.medium) * self.stiffness * normalized * 0.1
        retraction = -relu(self.far - length, inplace=True) * self.stiffness * normalized
        
        force = torch.where(sign == 1, attraction, retraction)
        force = torch.where(sign == 0, 0, force)
        return force.T

    def __repr__(self) -> str:
        return super().__repr__() + f'({self.stiffness})'