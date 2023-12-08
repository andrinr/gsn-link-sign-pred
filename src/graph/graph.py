from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from typing import NamedTuple
import jax.numpy as jnp
import torch
import jax
import graph as g
import matplotlib.pyplot as plt

class Measures(NamedTuple):
    max : int
    avg : int
    percentile : int
    values : jnp.ndarray

    def init(self, values : jnp.ndarray):
        self.max = jnp.max(values)
        self.avg = jnp.mean(values)
        self.percentile = jnp.percentile(values, 90)
        self.values = values
        return self

class SignedGraph(NamedTuple):
    """
    A signed directed graph.
    """
    edge_index : jnp.ndarray
    sign : jnp.ndarray
    sign_one_hot : jnp.ndarray
    degree : Measures
    centrality : Measures
    num_nodes : int
    num_edges : int
    test_mask : torch.Tensor
    train_mask : torch.Tensor

def to_SignedGraph(
    data : Data,
    reindexing : bool = True) -> SignedGraph:

    data, train_mask, test_mask = g.permute_split(data, 0.8)

    if reindexing:
        keep = torch.unique(data.edge_index)
        edge_index, edge_attr = subgraph(
            keep,
            data.edge_index,
            data.edge_attr,
            relabel_nodes=True,
        )
        data.edge_index = edge_index
        data.edge_attr = edge_attr

    edge_index = jnp.array(data.edge_index)
    signs = jnp.array(data.edge_attr)
    signs_one_hot = jax.nn.one_hot(signs + 1, 3)
    test_mask = jnp.array(test_mask)
    train_mask = jnp.array(train_mask)

    node_degrees = jnp.bincount(edge_index[0])

    degree_measures = Measures.init(node_degrees)

    num_nodes = jnp.max(edge_index) + 1
    num_edges = edge_index.shape[1]

    centrality = node_degrees
    centrality = centrality / degree_measures.average
    for i in range(3):
        edge_centrality = centrality[edge_index[1]]
        centrality = centrality.at[edge_index[0]].add(edge_centrality)
        centrality = centrality / degree_measures.average

    centrality_measures = Measures.init(centrality)

    #plot deg and centralities
    # two subplots
    fig, axs = plt.subplots(2)
    fig.suptitle('Degree and centrality distribution')
    axs[0].hist(node_degrees, histtype='step', bins=100)
    axs[0].set_title('Degree')
    axs[1].hist(centrality, histtype='step', bins=100)
    axs[1].set_title('Centrality')
    plt.show()

    node_degrees = jnp.expand_dims(node_degrees, axis=1)

    return SignedGraph(
        edge_index, 
        signs, 
        signs_one_hot,
        degree_measures,
        centrality_measures,
        num_nodes, 
        num_edges,
        test_mask,
        train_mask)
