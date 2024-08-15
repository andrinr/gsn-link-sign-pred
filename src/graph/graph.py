from torch_geometric.data import Data
from typing import NamedTuple
import jax.numpy as jnp
import graph as g
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
import torch
from torch_geometric.utils import subgraph

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
    A signed undirected graph.
    """
    num_nodes : int
    train_num_edges : int
    test_num_edges : int
    train_edge_index : jnp.ndarray
    test_edge_index : jnp.ndarray
    train_sign : jnp.ndarray
    test_sign : jnp.ndarray
    train_deg : Measures
    train_neg_deg : Measures
    train_pos_deg : Measures
    train_centr : Measures

def to_SignedGraph(
    data : Data,
    treat_as_undirected : bool) -> SignedGraph:

    transform = T.Compose([T.LargestConnectedComponents(num_components=1)])
    data = transform(data)

    # get node indices of largest connected component
    nodes = torch.unique(data.edge_index.flatten())

    edge_index, edge_attr  = subgraph(
        nodes, 
        data.edge_index,
        data.edge_attr,
        relabel_nodes=True)
    
    data = Data(edge_attr=edge_attr, edge_index=edge_index)

    transform = T.Compose([T.ToUndirected(reduce='min')])
    data = transform(data)

    train_data, test_data = g.permute_split(data, 0.8, treat_as_undirected)

    test_edge_index = jnp.array(test_data.edge_index)
    train_edge_index = jnp.array(train_data.edge_index)

    train_signs = jnp.array(train_data.edge_attr)
    test_signs = jnp.array(test_data.edge_attr)
    
    num_nodes = train_data.num_nodes

    test_num_edges = test_edge_index.shape[1]
    train_num_edges = train_edge_index.shape[1]

    train_degs = jnp.bincount(train_edge_index[0]) + 1

    print(f" avg deg {train_degs.mean()}")

    train_centr = train_degs
    train_centr = train_centr / jnp.max(train_centr)
    for i in range(3):
        edge_centrality = train_centr[train_edge_index[1]]
        train_centr = train_centr.at[train_edge_index[0]].add(edge_centrality)
        train_centr = train_centr / jnp.max(train_centr)

    #centrality = jnp.expand_dims(centrality, axis=1)
    centrality_measures = init_measures(jnp.expand_dims(train_centr, axis=1))

    train_centr = jnp.minimum(train_centr, centrality_measures.percentile) / centrality_measures.percentile
    centrality_measures = centrality_measures._replace(values=jnp.expand_dims(train_centr, axis=1))

    node_neg_degrees = jnp.zeros(train_degs.shape)
    node_neg_degrees = node_neg_degrees.at[train_edge_index[0]].add(train_signs < 0)
    node_neg_degrees = node_neg_degrees / train_degs
    neg_degree_measures = init_measures(jnp.expand_dims(node_neg_degrees, axis=1))

    node_pos_degrees = jnp.zeros(train_degs.shape)
    node_pos_degrees = node_pos_degrees.at[train_edge_index[0]].add(train_signs > 0)
    node_pos_degrees = node_pos_degrees / train_degs
    pos_degree_measures = init_measures(jnp.expand_dims(node_pos_degrees, axis=1))

    train_degs = jnp.expand_dims(train_degs, axis=1)
    degree_measures = init_measures(train_degs)

    train_degs = jnp.minimum(train_degs, degree_measures.percentile) / degree_measures.percentile
    degree_measures = degree_measures._replace(values=train_degs)

    print(train_edge_index.shape)
    print(train_signs.shape)
    print(test_edge_index.shape)
    print(test_signs.shape)
    print(num_nodes)
    print(train_num_edges)
    print(test_num_edges)
    print(degree_measures.values.shape)

    return SignedGraph(
        test_edge_index=test_edge_index,
        train_edge_index=train_edge_index,
        num_nodes=num_nodes,
        train_num_edges=train_num_edges,
        test_num_edges=test_num_edges,
        train_sign=train_signs,
        test_sign=test_signs,
        train_deg=degree_measures,
        train_neg_deg=neg_degree_measures,
        train_pos_deg=pos_degree_measures,
        train_centr=centrality_measures
    )
