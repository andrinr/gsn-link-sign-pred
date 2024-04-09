from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from typing import NamedTuple
import jax.numpy as jnp
import torch
import jax
import graph as g
import matplotlib.pyplot as plt

EPSILON = 1e-6

class Measures(NamedTuple):
    max : int
    avg : int
    percentile : int
    values : jnp.ndarray

def init_measures(values : jnp.ndarray):
    max = jnp.max(values)
    avg = jnp.mean(values)
    percentile = jnp.percentile(values, 80)

    return Measures(max, avg, percentile, values)

class SignedGraph(NamedTuple):
    """
    A signed directed graph.
    """
    edge_index : jnp.ndarray
    sign : jnp.ndarray
    sign_one_hot : jnp.ndarray
    degree_in : Measures
    neg_degree_in : Measures
    pos_degree_in : Measures
    degree_out : Measures
    neg_degree_out : Measures
    pos_degree_out : Measures
    num_nodes : int
    num_edges : int
    test_mask : torch.Tensor
    train_mask : torch.Tensor

def to_SignedGraph(
    data : Data,
    treat_as_undirected : bool,
    reindexing : bool = True) -> SignedGraph:

    data, train_mask, test_mask = g.permute_split(data, 0.8, treat_as_undirected)

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

    signs_train = jnp.where(train_mask, signs, 0)

    node_degrees_in = jnp.bincount(edge_index[0]) + EPSILON
    node_degrees_out = jnp.bincount(edge_index[1]) + EPSILON
    node_degrees = node_degrees_in + node_degrees_out

    # centrality = node_degrees
    # centrality = centrality / jnp.max(centrality)
    # for i in range(3):
    #     edge_centrality = centrality[edge_index[1]]
    #     centrality = centrality.at[edge_index[0]].add(edge_centrality)
    #     centrality = centrality / jnp.max(centrality)

    # #centrality = jnp.expand_dims(centrality, axis=1)
    # centrality_measures = init_measures(jnp.expand_dims(centrality, axis=1))

    # centrality = jnp.minimum(centrality, centrality_measures.percentile) / centrality_measures.percentile
    # centrality_measures = centrality_measures._replace(values=jnp.expand_dims(centrality, axis=1))

    node_neg_degrees_in = jnp.zeros(node_degrees_in.shape)
    node_neg_degrees_in = node_neg_degrees_in.at[edge_index[0]].add(signs_train < 0)
    node_neg_degrees_in = node_neg_degrees_in / node_degrees_in
    node_neg_degrees_in = jnp.where(node_degrees_in == 0, 0, node_neg_degrees_in)
    neg_degree_measures_in = init_measures(jnp.expand_dims(node_neg_degrees_in, axis=1))

    node_pos_degrees_in = jnp.zeros(node_degrees_in.shape)
    node_pos_degrees_in = node_pos_degrees_in.at[edge_index[0]].add(signs_train > 0)
    node_pos_degrees_in = node_pos_degrees_in / node_degrees_in
    node_pos_degrees_in = jnp.where(node_degrees_in == 0, 0, node_pos_degrees_in)
    pos_degree_measures_in = init_measures(jnp.expand_dims(node_pos_degrees_in, axis=1))

    node_neg_degrees_out = jnp.zeros(node_degrees_out.shape)
    node_neg_degrees_out = node_neg_degrees_out.at[edge_index[1]].add(signs_train < 0)
    node_neg_degrees_out = node_neg_degrees_out / node_degrees_out
    node_neg_degrees_out = jnp.where(node_degrees_out == 0, 0, node_neg_degrees_out)
    neg_degree_measures_out = init_measures(jnp.expand_dims(node_neg_degrees_out, axis=1))

    node_pos_degrees_out = jnp.zeros(node_degrees_out.shape)
    node_pos_degrees_out = node_pos_degrees_out.at[edge_index[1]].add(signs_train > 0)
    node_pos_degrees_out = node_pos_degrees_out / node_degrees_out
    node_pos_degrees_out = jnp.where(node_degrees_out == 0, 0, node_pos_degrees_out)
    pos_degree_measures_out = init_measures(jnp.expand_dims(node_pos_degrees_out, axis=1))

    num_nodes = jnp.max(edge_index) + 1
    num_edges = edge_index.shape[1]

    node_degrees_in = jnp.expand_dims(node_degrees_in, axis=1)
    degree_measures_in = init_measures(node_degrees_in)

    node_degrees_out = jnp.expand_dims(node_degrees_out, axis=1)
    degree_measures_out = init_measures(node_degrees_out)

    # node_degrees = jnp.minimum(node_degrees, degree_measures.percentile) / degree_measures.percentile
    # degree_measures = degree_measures._replace(values=node_degrees)

    return SignedGraph(
        edge_index, 
        signs, 
        signs_one_hot,
        degree_measures_in,
        neg_degree_measures_in,
        pos_degree_measures_in,
        degree_measures_out,
        neg_degree_measures_out,
        pos_degree_measures_out,
        num_nodes, 
        num_edges,
        test_mask,
        train_mask)
