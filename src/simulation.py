import networkx as nx
import numpy as np

def fast_chung_lung(degrees : np.ndarray):
    n_edges = np.sum(degrees) / 2
    n_nodes = len(degrees)

    print(n_edges)
    print(n_nodes)

    # probability matrix by multiplying degree vectors
    prob_matrix = np.outer(degrees, degrees) / (n_edges ** 2)
    prob_matrix_flat = prob_matrix.flatten()
    
    print(prob_matrix)
    print(np.sum(prob_matrix))
    # new empty graph
    G = nx.Graph()

    # m edge insertions
    for i in range(n_edges):
        # choose a random edge
        ind = np.random.choice(
            n_nodes ** 2, 
            replace=False, 
            p=prob_matrix_flat)

        u, v = np.unravel_index(ind, prob_matrix.shape)

        # add edge to graph
        G.add_edge(u, v)
    
    return G

fast_chung_lung(np.array([1, 2, 3, 4, 4]))
