import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from diffusion import graph_diffusion
from generators import random_edge_features
from loader import load_graph

graph = load_graph('datasets/slashdot.csv')
diffusion_series = graph_diffusion(graph, 50)


