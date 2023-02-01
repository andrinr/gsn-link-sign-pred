import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn.models import Node2Vec
from model import MassSpring
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class SpringTransform(BaseTransform):
    def __init__(
        self,
        device,
        embedding_dim: int,
        time_step : float,
        stiffness : float,
        damping : float,
        friend_distance : float,
        enemy_distance : float,
        noise : float,
        iterations : int,
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        self.time_step = time_step
        self.stiffness = stiffness
        self.damping = damping
        self.friend_distance = friend_distance
        self.enemy_distance = enemy_distance
        self.noise = noise
        self.iterations = iterations
        if self.device is None:
            self.compute_force = MassSpring(
                self.stiffness,
                self.enemy_distance,
                (self.enemy_distance - self.friend_distance) / 2,
                self.friend_distance).to(self.device)
        else:
            self.compute_force = MassSpring(
                self.stiffness,
                self.enemy_distance,
                (self.enemy_distance - self.friend_distance) / 2,
                self.friend_distance)

    def __call__(self, data: Data) -> Data:
        
        if self.device is not None:
            data = data.to(self.device)
            pos = torch.rand((data.num_nodes, self.embedding_dim), device=self.device)
            vel = torch.rand((data.num_nodes, self.embedding_dim), device=self.device)
        else:
            pos = torch.rand((data.num_nodes, self.embedding_dim))
            vel = torch.rand((data.num_nodes, self.embedding_dim))
            
        signs = data.edge_attr

        for i in tqdm(range(self.iterations)):
            force = self.compute_force(pos, data.edge_index, signs)
     
            # Symplectic Euler integration
            vel = vel * (1. - self.damping) + self.time_step * force
            pos = pos + self.time_step * vel
            # total force
            if i % 100 == 0:
                print(torch.norm(force, dim=1).mean().item())
                
        data.x = pos
        if self.device is None:
            return data
        else:
            return data.to('cpu')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'