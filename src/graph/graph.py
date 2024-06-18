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

def init_measures(values : jnp.ndarray, normalize : bool = False, log : bool = False) -> Measures:
    max = jnp.max(values)
    avg = jnp.mean(values)
    percentile = jnp.percentile(values, 80)

    if normalize:
        values = values / max

    if log:
        values = jnp.log(values)

    return Measures(max, avg, percentile, values)

def remove_nans(values : jnp.ndarray) -> jnp.ndarray:
    return jnp.nan_to_num(values, nan=0.0)

class SignedGraph(NamedTuple):
    """
    A signed undirected graph.
    """
    edge_index : jnp.ndarray
    sign : jnp.ndarray
    degree : Measures
    out_deg : Measures
    in_deg : Measures
    out_neg : Measures
    out_pos : Measures
    in_neg : Measures
    in_pos : Measures
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
    test_mask = jnp.array(test_mask)
    train_mask = jnp.array(train_mask)

    signs_train = jnp.where(train_mask, signs, 0)

    degrees_out = jnp.bincount(edge_index[0])
    degrees_in = jnp.bincount(edge_index[1])
    degree = degrees_out + degrees_in

    centrality = degree
    centrality = centrality / jnp.max(centrality)
    for i in range(3):
        edge_centrality = centrality[edge_index[1]]
        centrality = centrality.at[edge_index[0]].add(edge_centrality)
        centrality = centrality / jnp.max(centrality)

    #centrality = jnp.expand_dims(centrality, axis=1)
    centrality_measures = init_measures(jnp.expand_dims(centrality, axis=1))

    centrality = jnp.minimum(centrality, centrality_measures.percentile) / centrality_measures.percentile
    centrality_measures = centrality_measures._replace(values=jnp.expand_dims(centrality, axis=1))

    out_neg = jnp.zeros(degrees_out.shape)
    out_neg = out_neg.at[edge_index[0]].add(signs_train < 0)
    out_neg = out_neg / degrees_out
    out_neg = jnp.expand_dims(out_neg, axis=1)
    out_neg = remove_nans(out_neg)
    out_neg_measures = init_measures(out_neg)

    out_pos = jnp.zeros(degrees_out.shape)
    out_pos = out_pos.at[edge_index[0]].add(signs_train > 0)
    out_pos = out_pos / degrees_out
    out_pos = jnp.expand_dims(out_pos, axis=1)
    out_pos = remove_nans(out_pos)
    out_pos_measures = init_measures(out_pos)

    in_neg = jnp.zeros(degrees_in.shape)
    in_neg = in_neg.at[edge_index[1]].add(signs_train < 0)
    in_neg = in_neg / degrees_in
    in_neg = jnp.expand_dims(in_neg, axis=1)
    in_neg = remove_nans(in_neg)
    in_neg_measures = init_measures(in_neg)

    in_pos = jnp.zeros(degrees_in.shape)
    in_pos = in_pos.at[edge_index[1]].add(signs_train > 0)
    in_pos = in_pos / degrees_in
    in_pos = jnp.expand_dims(in_pos, axis=1)
    in_pos = remove_nans(in_pos)
    in_pos_measures = init_measures(in_pos)

    num_nodes = jnp.max(edge_index) + 1
    num_edges = edge_index.shape[1]

    degrees = jnp.expand_dims(degree, axis=1)
    degrees = init_measures(degrees, log=True)

    degrees_out = degrees_out / degree
    degrees_out = remove_nans(degrees_out)  
    degrees_out = jnp.expand_dims(degrees_out, axis=1)
    degrees_out = init_measures(degrees_out, log=False)

    degrees_in = degrees_in / degree
    degrees_in = remove_nans(degrees_in)
    degrees_in = jnp.expand_dims(degrees_in, axis=1)
    degrees_in = init_measures(degrees_in, log=False)

    graph = SignedGraph(
        edge_index, 
        signs, 
        degrees,
        degrees_out,
        degrees_in,
        out_neg_measures,
        out_pos_measures,
        in_neg_measures,
        in_pos_measures,
        centrality_measures,
        num_nodes, 
        num_edges,
        test_mask,
        train_mask)

    return graph
