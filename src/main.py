import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.sampler import NeighborSampler

from diffusion import graph_diffusion
from generators import random_edge_features
from loader import load_graph

graph = load_graph('datasets/slashdot.csv')
sampler = NeighborSampler(graph, num_neighbors=100)


loader = NeighborLoader(
    graph,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=128,
    input_nodes=data.train_mask,
)

for 
sampled_data = next(iter(loader))
print(sampled_data.batch_size)

diffusion_series = graph_diffusion(graph, 50)




