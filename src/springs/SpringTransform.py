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

        self.started = False
        
        self.compute_force = MassSpring(
            self.enemy_distance,
            self.enemy_stiffness,
            self.neutral_distance,
            self.neutral_stiffness,
            self.friend_distance,
            self.friend_stiffness)
            
    def __call__(self, data: Data) -> Data:

        if not self.started:
            torch.manual_seed(self.seed)
        
            self.pos = torch.rand((data.num_nodes, self.embedding_dim), device=data.edge_index.device) * 2.0 - 1.0
            self.vel = torch.zeros((data.num_nodes, self.embedding_dim), device=data.edge_index.device)
           
            self.pos *= 5.0

            self.started = True

        self.start_iteration = 0

        signs = data.edge_attr

        pbar = tqdm(range(self.start_iteration, self.start_iteration + self.iterations))
        energies = []
        dt = self.time_step
        dt2 = self.time_step * self.time_step
        for i in pbar:
            random_noise = torch.randn((data.num_nodes, self.embedding_dim), device=data.edge_index.device)

            self.vel = self.vel + random_noise * dt * (1.0 - (i / self.iterations)) * 0.01

            if self.integration_scheme == 'symplectic_euler':
                # Symplectic Euler integration
                force = self.compute_force(self.pos, data.edge_index, signs)
                self.vel = self.vel + dt * self.force
                self.pos = self.pos + dt * self.vel

            elif self.integration_scheme == 'verlet':
                # Verlet integration
                force_a = self.compute_force(self.pos, data.edge_index, signs)
                self.pos = self.pos + dt * self.vel + 0.5 * dt2 * force_a
                force_b = self.compute_force(self.pos, data.edge_index, signs)
                self.vel = self.vel + 0.5 * dt * (force_a + force_b)

            elif self.integration_scheme == 'leapfrog':
                # Leapfrog integration
                force = self.compute_force(self.pos, data.edge_index, signs)
                self.vel = self.vel + 0.5 * dt * force
                self.pos = self.pos + dt * self.vel
                force = self.compute_force(self.pos, data.edge_index, signs)
                self.vel = self.vel + 0.5 * dt * force
                
            energy = torch.sqrt(torch.norm(self.vel, dim=1, keepdim=False)) + torch.norm(force, dim=1, keepdim=False)
            self.energy_total = torch.sum(energy)


            pbar.set_description(f"Energy: {self.energy_total.item():.2f}")
          
            self.vel = self.vel * (1.0 - self.damping * (i / self.iterations))

        data.x = self.pos

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'