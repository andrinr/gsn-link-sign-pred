import networkx as nx
import numpy as np
def simulate(n_nodes : int, n_steps : int, n_opinions, p_init : float = 0.1):

    G = nx.empty_graph(n_nodes)

    # init opinions randomly
    random = np.random.random((n_dims, n_nodes))
    indices = np.arange(n_nodes)

    for i in range(n_opinions):
        nx.set_node_attributes(G, dict(zip(indices, random[i])), f"dim_{i}")

    
    for i in range(n_steps):
        likelihoods = np.array(list(dict(G.degree()).values())) + 1
        

simulate(10, 10, 3)