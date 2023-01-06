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

graph = load_graph('datasets/slashdot.csv')
samples = sample(graph, 100, 1000)

nx.draw(to_networkx(samples[0]))

diffusion_series = graph_diffusion(graph, 50)
