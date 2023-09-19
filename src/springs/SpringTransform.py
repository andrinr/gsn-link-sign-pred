import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree
from springs import MassSpring
from tqdm import tqdm

class SpringTransform(BaseTransform):
    """
    SpringTransform

    Parameters:
    ----------
    - device : torch.device
        Device to use
    - embedding_dim : int
        Embedding dimension
    - time_step : float
        Time step
    - damping : float
        Damping
    - friend_distance : float 
        Friend distance
    - friend_stiffness : float
        Friend stiffness
    - neutral_distance : float
        Neutral distance
    - neutral_stiffness : float
        Neutral stiffness
    - enemy_distance : float
        Enemy distance
    """
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
        integration_scheme : str = 'leapfrog',
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
        self.integration_scheme = integration_scheme
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
        
        pos *= 5.0
        signs = data.edge_attr

        node_degrees = degree(data.edge_index[0], data.num_nodes, dtype=torch.float)

        mass = node_degrees * 0.3 + 1

        pbar = tqdm(range(self.iterations))
        energies = []
        dt = self.time_step
        dt2 = self.time_step * self.time_step
        for i in pbar:
            
            if self.integration_scheme == 'symplectic_euler':
                # Symplectic Euler integration
                force = self.compute_force(pos, data.edge_index, signs)
                vel = vel + dt * force / mass
                pos = pos + dt * vel

            elif self.integration_scheme == 'verlet':
                # Verlet integration
                force_a = self.compute_force(pos, data.edge_index, signs)
                pos = pos + dt * vel + 0.5 * dt2 * force_a
                force = self.compute_force(pos, data.edge_index, signs)
                vel = vel + 0.5 * dt * (force + force)

            elif self.integration_scheme == 'leapfrog':
                # Leapfrog integration
                force = self.compute_force(pos, data.edge_index, signs)
                vel = vel + 0.5 * dt * force
                pos = pos + dt * vel
                force = self.compute_force(pos, data.edge_index, signs)
                vel = vel + 0.5 * dt * force
                
            energy = torch.sqrt(torch.norm(vel, dim=1, keepdim=False)) + torch.norm(force, dim=1, keepdim=False)
            energy_total = torch.sum(energy)

            energies.append(energy_total.item())

            pbar.set_description(f"Energy: {energy_total.item():.2f}")
            damping_force = self.damping * torch.norm(vel, dim=1, keepdim=True)
            vel = vel * (1. - self.damping) * torch.norm(vel, dim=1, keepdim=True)
            vel = vel * node_degrees

        data.x = pos

        self.energies = energies

        self.final_per_node_energy = energy
        self.final_per_node_vel = torch.norm(vel, dim=1, keepdim=False)
        self.final_per_node_force = torch.norm(force, dim=1, keepdim=False)
        self.final_per_node_pos = torch.norm(pos, dim=1, keepdim=False)

        if self.device is None:
            return data
        else:
            return data.to('cpu')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'