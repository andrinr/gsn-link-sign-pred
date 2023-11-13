from torch_geometric.data import Data 
from typing import NamedTuple
import jax.numpy as jnp
from graph import permute_split
import matplotlib.pyplot as plt

class SignedGraph(NamedTuple):
    edge_index : jnp.ndarray
    sign : jnp.ndarray
    node_degrees : jnp.ndarray
    num_nodes : int
    num_edges : int
    train_mask : jnp.ndarray
    test_mask : jnp.ndarray
    val_mask : jnp.ndarray

    
def to_SignedGraph(data : Data) -> SignedGraph:
    data, train_mask, val_mask, test_mask = permute_split(data, 0.1, 0.8)

    num_nodes = data.num_nodes
    num_edges = data.num_edges

    train_mask = jnp.array(train_mask)
    val_mask = jnp.array(val_mask)
    test_mask = jnp.array(test_mask)

    # convert to jnp arrays from torch tensors
    edge_index = jnp.array(data.edge_index)
    signs = jnp.array(data.edge_attr)

    node_degrees = jnp.bincount(edge_index[0])

    return SignedGraph(
        edge_index, 
        signs, 
        node_degrees, 
        num_nodes, 
        num_edges, 
        train_mask,
        test_mask, 
        val_mask)