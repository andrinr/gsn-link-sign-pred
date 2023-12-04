from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from typing import NamedTuple
import jax.numpy as jnp
import torch
import jax
import graph as g

class SignedGraph(NamedTuple):
    """
    A signed directed graph.
    """
    edge_index : jnp.ndarray
    sign : jnp.ndarray
    sign_one_hot : jnp.ndarray
    node_degrees : jnp.ndarray
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
    node_degrees = jnp.expand_dims(node_degrees, axis=1)

    num_nodes = jnp.max(edge_index) + 1
    num_edges = edge_index.shape[1]

    return SignedGraph(
        edge_index, 
        signs, 
        signs_one_hot,
        node_degrees,
        num_nodes, 
        num_edges,
        test_mask,
        train_mask)
