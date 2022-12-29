import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt

p = 0.1
N = 30

def gen_adjacency_matrix(N : int, edge_prob : float, positive_prob : float = 0.5):
    rd = np.random.random((N, N))
    A = rd < p
    #A = A + A.T
    return A

A = gen_adjacency_matrix(N, 0.1, 0.8)
print(A)
G = nx.from_numpy_matrix(A)

bb = nx.edge_betweenness_centrality(G, normalized=False)
print(bb)

nx.draw(G, with_labels=True)
plt.show()