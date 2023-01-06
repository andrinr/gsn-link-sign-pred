import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Source: Signed Network Modeling Based on Structural Balance Theory
def BSCL(degrees : np.ndarray, p_pos : float = 0.5):
    G = fast_chung_lung(degrees)
    G = sign_partition(G, p_pos)

    n_edges = G.number_of_edges()

    node_choices = np.random.choice(degrees, n_edges * 2, replace=True)
    for i in range(n_edges):
       
    
    return G

def sign_partition(G : nx.graph, p_pos : float = 0.5):
    p_neg = 1 - p_pos
    random_signs = np.random.choice([-1, 1], G.number_of_nodes(), p=[p_neg, p_pos])
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['sign'] = random_signs[i] 

    return G

# Source: GENERATING LARGE SCALE-FREE NETWORKS WITH THE CHUNG–LU RANDOM GRAPH MODEL∗
def fast_chung_lung(degrees : np.ndarray):
    n_edges = np.sum(degrees) / 2
    if n_edges != int(n_edges): raise ValueError("degrees must be even")
    n_edges = int(n_edges)

    n_nodes = len(degrees)

    # probability matrix by multiplying degree vectors
    prob_matrix = np.outer(degrees, degrees) / ((n_edges * 2) ** 2)
    prob_matrix_flat = prob_matrix.flatten()
    
    if np.sum(prob_matrix) != 1: raise ValueError("prob_matrix must sum to 1")
    # new empty graph with self loops
    G = nx.empty_graph(n_nodes, create_using=nx.DiGraph)

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

G = fast_chung_lung(np.array([1, 3, 4, 4, 10, 100]))
nx.draw(G, with_labels=True)
plt.show()