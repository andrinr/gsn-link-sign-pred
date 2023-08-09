import torch
from torch_geometric.nn import MessagePassing
from torch.nn.functional import relu

class MassSpring(MessagePassing):
    def __init__(self, 
        enemy_distance : float,
        enemy_stiffness : float,
        neutral_distance : float,
        neutral_stiffness : float,
        friend_distance : float,
        friend_stiffness : float):
        super().__init__(aggr='add') 
        self.enemy_distance = enemy_distance
        self.enemy_stiffness = enemy_stiffness
        self.neutral_distance = neutral_distance
        self.neutral_stiffness = neutral_stiffness
        self.friend_distance = friend_distance
        self.friend_stiffness = friend_stiffness

    def forward(self, position, edge_index, sign):
        return self.propagate(edge_index, position=position, sign=sign)

    def message(self, position_i, position_j, sign):
        spring = position_j - position_i
        length = torch.norm(spring, dim=1, keepdim=False)
        normalized = torch.div(spring.T, length + 0.001)
        attraction = relu(length - self.friend_distance) * self.friend_stiffness * normalized
        regular = (length - self.neutral_distance) * self.neutral_stiffness * normalized
        retraction = -relu(self.enemy_distance - length) * self.enemy_stiffness * normalized
        
        force = torch.where(sign == 1, attraction, retraction)
        force = torch.where(sign == 0, regular, force)
        return force.T