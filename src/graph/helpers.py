from torch_geometric.data import Data
import numpy as np
from torch_geometric.utils import to_networkx, remove_self_loops
from networkx.algorithms.cycles import simple_cycles

def check_edge_exists(data : Data, u : int, v : int):
    edge_index = data.edge_index
    src, dst = edge_index
    node_edges = src == u
    node_edges &= dst == v
    return node_edges.any()

def random_hop(data : Data, u : int):
    neighbors = get_neighbors(data, u)
    if len(neighbors) == 0:
        return None
    v = np.random.choice(neighbors)
    return v

def get_neighbors(data : Data, u : int):
    src, dst = data.edge_index
    node_edges = src == u
    return dst[node_edges]

def get_edge_index(data : Data, u, v):
    edge_index = data.edge_index
    src, dst = edge_index
    node_edges = src == u
    node_edges &= dst == v
    index = np.where(node_edges)[0]

    return index

def get_cycles(data : Data, degree : int):
    G = to_networkx(data)
    cycles = simple_cycles(G, length_bound=degree)
    
    # get signs of cycles
    cycles = list(cycles)
    cycle_signs = []
    # init list of lists
    node_cycles = [[] for _ in range(data.num_nodes)]

    for cycle in cycles:
        sign = 1
        deg = len(cycle)
        for i in range(deg):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            index = get_edge_index(data, u, v)
            sign *= data.edge_attr[index]
        
        node_cycles[u].append((deg, sign))
        node_cycles[v].append((deg, sign))

    return node_cycles