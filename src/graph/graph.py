from torch_geometric.data import Data
from torch_geometric.utils.map import map_index
from torch_geometric.utils import subgraph
from typing import NamedTuple
import jax.numpy as jnp
from graph import permute_split
import torch

class SignedGraph(NamedTuple):
    edge_index : jnp.ndarray
    sign : jnp.ndarray
    node_degrees : jnp.ndarray
    num_nodes : int
    num_edges : int
    train_mask : jnp.ndarray
    test_mask : jnp.ndarray
    val_mask : jnp.ndarray

    
def to_SignedGraph(
    data : Data,
    reindexing : bool = True) -> SignedGraph:
    data, train_mask, val_mask, test_mask = permute_split(data, 0.1, 0.8)

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

    train_mask = jnp.array(train_mask)
    val_mask = jnp.array(val_mask)
    test_mask = jnp.array(test_mask)

    # convert to jnp arrays from torch tensors
    edge_index = jnp.array(data.edge_index)
    signs = jnp.array(data.edge_attr)
    node_degrees = jnp.bincount(edge_index[0])

    num_nodes = jnp.max(edge_index) + 1
    num_edges = edge_index.shape[1]

    return SignedGraph(
        edge_index, 
        signs, 
        node_degrees, 
        num_nodes, 
        num_edges, 
        train_mask,
        test_mask, 
        val_mask)