import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from model import MassSpring
from tqdm import tqdm
from torch_geometric.utils import degree

class SpringTransform(BaseTransform):
    def __init__(
        self,
        device,
        embedding_dim: int,
        time_step : float,
        damping : float,
        friend_distance : float,
        friend_stiffness : float,
        neutral_distance : float,
        neutral_stiffness : float,
        enemy_distance : float,
        enemy_stiffness : float,
        iterations : int,
        seed : int = 1
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        self.time_step = time_step
        self.damping = damping
        self.friend_distance = friend_distance
        self.friend_stiffness = friend_stiffness
        self.neutral_distance = neutral_distance
        self.neutral_stiffness = neutral_stiffness
        self.enemy_distance = enemy_distance
        self.enemy_stiffness = enemy_stiffness
        self.iterations = iterations
        self.seed = seed
        
        self.compute_force = MassSpring(
            self.enemy_distance,
            self.enemy_stiffness,
            self.neutral_distance,
            self.neutral_stiffness,
            self.friend_distance,
            self.friend_stiffness)

        if self.device is not None:
            self.compute_force = self.compute_force.to(device)
            
    def __call__(self, data: Data) -> Data:

        torch.manual_seed(self.seed)
        if self.device is not None:
            data = data.to(self.device)
            pos = torch.rand((data.num_nodes, self.embedding_dim), device=self.device) * 2.0 - 1.0
            vel = torch.zeros((data.num_nodes, self.embedding_dim), device=self.device)
        else:
            pos = torch.rand((data.num_nodes, self.embedding_dim)) * 2.0 - 1.0
            vel = torch.zeros((data.num_nodes, self.embedding_dim))
        
        pos *= 2.0
        signs = data.edge_attr
        
        pbar = tqdm(range(self.iterations))
        for i in pbar:
            force = self.compute_force(pos, data.edge_index, signs)
            # Symplectic Euler integration
            vel = vel * (1. - self.damping) + self.time_step * force
            pos = pos + self.time_step * vel
                
        data.x = pos

        if self.device is None:
            return data
        else:
            return data.to('cpu')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'