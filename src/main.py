import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt

from diffusion import diffuse
from generators import random_edge_features


features = random_edge_features(10, 0.1, 0.01)

steps = 10
fig, axs = plt.subplots(ncols=steps, figsize = (steps * 1, 3))

diffused_features = diffuse(features, steps)

for i in range(steps):
    G = nx.from_numpy_matrix(diffused_features[i])
    nx.draw_circular(G, with_labels=True, ax=axs[i], node_size = 10)
    axs[i].set_title(f"Step {i}")

plt.show()