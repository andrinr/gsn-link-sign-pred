import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

def graph_diffusion(graph : Data, steps : int):
    """"
    Take a pyg graph and return a diffusion seriess

    Parameters
    ----------
    graph : Data
        pyg graph
    steps : int
        number of steps to take

    Returns
    -------
    diffusion_series : list
        diffusion series
    """

    edge_index : torchTensor = graph.edge_index
    edge_attr : torchTensor = graph.edge_attr

    negative_fraction = np.count_nonzero(edge_attr == -1) / len(edge_attr)
    positive_fraction = 1.0 - negative_fraction

    #TODO: think about sampling from the distribution of edge_attr
    #degs = degree(edge_index[0])

    oversampling_factor = 1.1
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    num_edges_oversampled = int(num_edges * oversampling_factor)

    random_signs = np.random.choice(
        np.array([-1, 1], dtype=np.float32),
        size=(num_edges),
        p=[negative_fraction, positive_fraction]
    )
    #TODO: compute expected number of self loops and duplicates from random edge sampling
    # oversample and then remove self loops and duplicates

    # random edges with no self loops and no duplicates
    random_edges = np.random.randint(0, num_nodes, size=(num_edges_oversampled, 2))
    random_edges = np.unique(random_edges, axis=0)
    random_edges = random_edges[random_edges[:,0] != random_edges[:,1]]
    random_edges = random_edges[:num_edges]

    addition_time = np.random.random(size=num_edges) 

    random_edges = torch.from_numpy(random_edges.T)
    random_signs = torch.from_numpy(random_signs)
    addition_time = torch.from_numpy(addition_time)

    series = []
    for step in range(steps):
        fraction = 1.0 / steps * (step + 1)
        diffused_edge_index = torch.clone(edge_index)
        diffused_edge_index[:,addition_time < fraction] = random_edges[:,addition_time < fraction] 
        diffused_edge_attr = torch.clone(edge_attr)
        diffused_edge_attr[addition_time < fraction] = random_signs[addition_time < fraction]

        diffused_graph = Data(edge_index=diffused_edge_index, edge_attr=diffused_edge_attr)
        diffused_graph.coalesce()
        series.append(diffused_graph)

    return series

def adjacency_diffusion(edge_features : np.ndarray,  steps : int):
    # TODO: handle upper lower triangular matrix
    shape = edge_features.shape
    total = shape[0] * shape[1]
    negative_fraction = np.count_nonzero(edge_features == -1) / total
    positive_fraction = np.count_nonzero(edge_features == 1) / total
    neutral_fraction = 1.0 - (negative_fraction + positive_fraction)

    change_likelihood = np.random.random(shape)
    change_to = np.random.choice([-1, 0, 1], size=shape, p=[negative_fraction, neutral_fraction, positive_fraction])

    series = []
    # the very last step should be a more ore less random matrix with similar distribution of positive and negative values
    for step in range(steps):
        p_change = 1.0 / steps * (step + 1)

        change_mask = change_likelihood < p_change
        
        diffused_mat = edge_features.copy()
        diffused_mat[change_mask] = change_to[change_mask]

        series.append(diffused_mat)

    return series
