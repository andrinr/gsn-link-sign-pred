from torch_geometric.data import Data
from torch_geometric.utils.map import map_index
from torch_geometric.utils import subgraph
from typing import NamedTuple
import jax.numpy as jnp
from graph import permute_split
import torch

class SignedGraph(NamedTuple):
    """
    A signed directed graph.
    """
    edge_index : jnp.ndarray
    sign : jnp.ndarray
    node_degrees : jnp.ndarray
    num_nodes : int
    num_edges : int
    train_mask : jnp.ndarray
    test_mask : jnp.ndarray
    subsample_pos_mask : jnp.ndarray
    subsample_neg_mask : jnp.ndarray

    def get_nodes(self, edge_index : int) -> (int, int):
        a = self.edge_index.at[0, edge_index].get()
        b = self.edge_index.at[1, edge_index].get()

        return a, b
    
    def get_neighbors(self, node_index : int) -> jnp.ndarray:
        self.edge_index.at[0, self.edge_index.at[0, :] == node_index].get()

def to_SignedGraph(
    data : Data,
    reindexing : bool = True) -> SignedGraph:
    data, train_mask, val_mask, test_mask, num_train, num_val, num_test = permute_split(data, 0, 0.9)

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
    test_mask = jnp.array(test_mask)

    # mark first 1000 nodes in test mask as subsample mask
    number_of_negative_edges = jnp.sum(data.edge_attr == -1)
    subsample_size = min(1000, number_of_negative_edges)

    # find indices of negative edges
    negative_edge_indices = jnp.where(data.edge_attr == -1)[0]
    positive_edge_indices = jnp.where(data.edge_attr == 1)[0]

    # randomly select subsample_size indices
    subsample_pos_indices = jnp.random.choice(negative_edge_indices, subsample_size, replace=False)
    subsample_neg_indices = jnp.random.choice(positive_edge_indices, subsample_size, replace=False)

    # create subsample mask
    subsample_pos_mask = jnp.zeros(data.edge_attr.shape[0], dtype=bool)
    subsample_pos_mask = subsample_pos_mask.at[subsample_pos_indices].set(True)

    subsample_neg_mask = jnp.zeros(data.edge_attr.shape[0], dtype=bool)
    subsample_neg_mask = subsample_neg_mask.at[subsample_neg_indices].set(True)


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
        subsample_pos_mask,
        subsample_neg_mask)