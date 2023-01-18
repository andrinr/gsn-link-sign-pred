import numpy as np
import torch
from torch_geometric.data import Data

def node_sign_diffusion(node_features, fraction : float):

    if fraction > 1.0 or fraction < 0.0:
        raise ValueError("fraction should be between 0 and 1")
    n = node_features.shape[0]

    negative_fraction = (torch.count_nonzero(node_features == -1) / n).item()
    positive_fraction = 1.0 - negative_fraction

    random_signs = np.random.choice(
        np.array([0, 1], dtype=np.int32),
        size=(n,1),
        p=[negative_fraction, positive_fraction]
    )
    random_signs = torch.tensor(random_signs, dtype=torch.long)

    mutation_time = np.random.random((n, 1))
    mutation_time = torch.tensor(mutation_time, dtype=torch.long)

    diffused_node_features = torch.clone(node_features)
    diffused_node_features[mutation_time < fraction] = random_signs[mutation_time < fraction]

    return diffused_node_features

def sign_diffusion(graph, fraction : float):
    """
    Take a pyg gtaph and return a diffusion series where only signs are diffused
    """

    if fraction > 1.0 or fraction < 0.0:
        raise ValueError("fraction should be between 0 and 1")

    edge_index = graph.edge_index
    edge_attr = graph.edge_attr
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges

    negative_fraction = np.count_nonzero(edge_attr == -1) / len(edge_attr)
    positive_fraction = 1.0 - negative_fraction

    random_signs = np.random.choice(
        np.array([-1, 1], dtype=np.float32),
        size=(num_edges),
        p=[negative_fraction, positive_fraction]
    )

    muttation_time = np.random.random(size=num_edges) 

    diffused_edge_attr = torch.clone(edge_attr)
    diffused_edge_attr[muttation_time < fraction] = random_signs[muttation_time < fraction]

    diffused_graph = Data(edge_index=edge_attr, edge_attr=diffused_edge_attr)
    diffused_graph.coalesce()
    
    return diffused_graph

def sign_link_diffusion(graph : Data, fraction : float):
    """"
    Take a pyg graph and return a diffusion seriess where both signs and links are diffused
    """
    if fraction > 1.0 or fraction < 0.0:
        raise ValueError("fraction should be between 0 and 1")

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

    muttation_time = np.random.random(size=num_edges) 

    random_edges = torch.from_numpy(random_edges.T)
    random_signs = torch.from_numpy(random_signs)
    muttation_time = torch.from_numpy(muttation_time)

    diffused_edge_index = torch.clone(edge_index)
    diffused_edge_index[:,muttation_time < fraction] = random_edges[:,muttation_time < fraction] 
    diffused_edge_attr = torch.clone(edge_attr)
    diffused_edge_attr[muttation_time < fraction] = random_signs[muttation_time < fraction]

    diffused_graph = Data(edge_index=diffused_edge_index, edge_attr=diffused_edge_attr)
    diffused_graph.coalesce()
    
    return diffused_graph

def adjacency_diffusion(edge_features : np.ndarray,  steps : int):
    """
    Only works with dense adjecency matrices
    """
    
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
