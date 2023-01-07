import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.sampler import NeighborSampler
from torch_geometric.utils import to_networkx

from diffusion import graph_diffusion
from generators import random_edge_features
from loader import load_graph
from sampler import sample
from simulation.BSCL import BSCL
from helpers import even_uniform, even_exponential

G = BSCL(even_exponential(5, 100))
edges = G.edges()
pos = nx.spring_layout(G, seed=63)
colors = [G[u][v]['sign'] for u, v in edges]

nx.draw(G, pos, edge_color=colors, node_size=10)
plt.show()