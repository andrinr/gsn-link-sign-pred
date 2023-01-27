import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential, Linear, ReLU

class OpinionLayer(MessagePassing):
    def __init__(self, 
        pe_channels : int,
        hidden_channels : int,):
        super().__init__(aggr='add') 

        n_channels_in = pe_channels * 2 + 1 + hidden_channels * 2 + 1
        n_channels_hidden = n_channels_in
        n_channels_out = pe_channels + hidden_channels
        self.mlp = Sequential(
            Linear(n_channels_in, n_channels_hidden), 
            ReLU(), 
            Linear(n_channels_hidden, n_channels_out))
        
    def forward(self, position, hidden, edge_index, sign):
        return self.propagate(edge_index, position=position, hidden=hidden, sign=sign)

    def message(self, position_i, position_j, hidden_i, hidden_j, sign):
        vector = position_j - position_i
        length = torch.norm(vector, dim=1, keepdim=False)
        vector = torch.div(vector.T, length + 0.001)

        x = torch.cat((position_i, vector, length, hidden_i, hidden_j), dim=1)
        
        return self.mlp(x)

    def __repr__(self) -> str:
        return super().__repr__() + f'({self.stiffness})'