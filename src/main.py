import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt

from diffusion import diffuse

p = 0.1
N = 10

def gen_signed_adj_matrix(N : int, edge_prob : float, positive_prob : float = 0.5):
    rd = np.random.random((N, N))
    A = rd < edge_prob
    A_signed = A * np.random.choice([-1, 1], size=A.shape, p=[1 - positive_prob, positive_prob])
    return A, A_signed

A, A_signed = gen_signed_adj_matrix(N, 0.1, 0.8)

steps = 10
fig, axs = plt.subplots(ncols=steps, figsize = (steps * 1, 3))

diffused = diffuse(A_signed, steps - 1)

for i in range(steps):
    G = nx.from_numpy_matrix(diffused[i])
    nx.draw_circular(G, with_labels=True, ax=axs[i], node_size = 10)
    axs[i].set_title(f"Step {i}")

plt.show()