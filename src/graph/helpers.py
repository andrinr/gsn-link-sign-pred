from torch_geometric.data import Data
import numpy as np

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

def get_edge_index(data, u, v):
    edge_index = data.edge_index
    src, dst = edge_index
    node_edges = src == u
    node_edges &= dst == v
    index = np.where(node_edges)[0]

    return index

def to_directed(data):
    assert data.is_undirected()
    data.edge_index = data.edge_index[:, data.edge_index[0] <= data.edge_index[1]]

    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[data.edge_index[0] <= data.edge_index[1]]

    return data

    
