import networkx as nx
import numpy as np
from queue import PriorityQueue

def sample(G, n_samples):

    n_nodes = G.number_of_nodes()
    init_node = np.random.randint(n_nodes, size=n_samples)

    for i in range(n_samples):
        queue = []
        queue.put((0, init_node[i]))
        while len(visited) < n_nodes:
            node = q.get()
            queue.extend(G.neighbors(node))

            visited.add(next_node)
            init_node[i] = next_node
    visited = set()
    set.add(np.ranomd.randint(n_nodes))