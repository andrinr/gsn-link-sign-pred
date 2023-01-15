import numpy as np
import networkx as nx
import random
from typing import Any, Callable, Dict, List, Optional, Union
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.datasets.graph_generator import GraphGenerator

class BSCLGraph(GraphGenerator):
    def __init__(
        self, 
        degree_generator : Optional[Callable] = None,
        p_positive_sign : Optional[float] = 0.9,
        p_close_triangle : Optional[float] = 0.2,
        p_close_for_balance : Optional[float] = 0.8,
        remove_self_loops : Optional[bool] = True):

        self.degree_generator = degree_generator
        self.p_positive_sign = p_positive_sign
        self.p_close_triangle = p_close_triangle
        self.p_close_for_balance = p_close_for_balance
        self.remove_self_loops = remove_self_loops

    def __call__(self) -> Data:
        """
        Generates a signed graph with the given degrees and p_pos.
        Based on paper: Signed Network Modeling Based on Structural Balance Theory
        Structural Balance Theory: https://en.wikipedia.org/wiki/Balance_theory

        Args:
            degrees (np.ndarray): The degrees of the nodes in the graph.
            p_pos (float, optional): The probability of a node being positive. Defaults to 0.5.
        
        Returns:
            nx.graph: The generated graph.
        """

        degrees = self.degree_generator()
        G = fast_chung_lung(degrees)
        # return list of edges from edge view iterable
        old_edges_list = list(G.edges)
        random.shuffle(old_edges_list)
        G = sign_partition(G, self.p_positive_sign)

        n_edges = G.number_of_edges()
        n_nodes = G.number_of_nodes()

        # Precompute node choices for all iterations for performance
        probabilities = degrees / np.sum(degrees)
        probabilities[0] += 1.0 - np.sum(probabilities)
        node_choices = np.random.choice(
            n_nodes, 
            n_edges * 2, 
            p=degrees / np.sum(degrees),
            replace=True)

        for i in range(n_edges):
            u = node_choices[i]
            # close a triangle
            if coin(self.p_close_triangle):
                res = two_hop_walk(G, u)
                if not res: continue
                v, w = res
                sign = G[u][v]['sign'] * G[v][w]['sign']
                # make it balanced
                if coin(self.p_close_for_balance):
                    G.add_edge(u, w, sign=sign)
                # make it unbalanced
                else:
                    G.add_edge(u, w, sign=invert(sign))
            # insert random edge
            else:
                v = node_choices[i + n_edges]
                G.add_edge(u, v, sign=coin(self.p_positive_sign) * 2 - 1)

            a, b = old_edges_list[i]
            G.remove_edge(a, b)

        if self.remove_self_loops:
            G.remove_edges_from(nx.selfloop_edges(G))

        return from_networkx(G)

    def __repr__(self) -> str:
        return '{}(p_positive_sign={}, p_close_triangle={}, p_close_for_balance={}, remove_self_loops={})'.format(
            self.__class__.__name__,
            self.p_positive_sign,
            self.p_close_triangle,
            self.p_close_for_balance,
            self.remove_self_loops)


def two_hop_walk(G, u):
    """ 
    Performs a two hop walk on the graph G starting at node u.

    Args:
        G (nx.graph): The graph to perform the walk on.
        u (int): The node to start the walk at.

    Returns:
        tuple: The two nodes that were visited.
    """
    neighbors = list(G.neighbors(u))
    if len(neighbors) == 0:
        return None
    v = np.random.choice(neighbors)
    neighbors = list(G.neighbors(v))
    if len(neighbors) == 0:
        return None
    w = np.random.choice(neighbors)
    return v, w

def fast_chung_lung(degrees : np.ndarray):
    """
    Generates a graph with the given degrees.
    Based on paper: GENERATING LARGE SCALE-FREE NETWORKS WITH THE CHUNG–LU RANDOM GRAPH MODEL∗

    Args:
        degrees (np.ndarray): The degrees of the nodes in the graph.

    Returns:
        nx.graph: The generated graph.
    """

    n_edges = np.sum(degrees) / 2
    if n_edges != int(n_edges): raise ValueError("degrees must be even")
    n_edges = int(n_edges)
    n_nodes = len(degrees)

    # probability matrix by multiplying degree vectors
    prob_matrix = np.outer(degrees, degrees) / ((n_edges * 2) ** 2)
    prob_matrix_flat = prob_matrix.flatten()
    # avoid numerical errors, we offset maximal probability by numerical rounding error
    # maximal entry is choosen to avoid negative probabilites
    max_index = np.argmax(prob_matrix_flat)
    prob_matrix_flat[max_index] += 1.0 - np.sum(prob_matrix_flat)
    
    G = nx.empty_graph(n_nodes)

    # choose random edges according to the probability matrix
    ind = np.random.choice(
        n_nodes ** 2,
        n_edges,
        replace=False, 
        p=prob_matrix_flat)

    # add the random edges
    u, v = np.unravel_index(ind, prob_matrix.shape)
    for i in range(n_edges):
        G.add_edge(u[i], v[i])
    
    return G

def is_triad_balanced(G : nx.graph, u : int, v : int, w : int):
    return triad_sign == 1

def triad_sign(G : nx.graph, u : int, v : int, w : int):
    return G.nodes[u]['sign'] * G.nodes[v]['sign'] * G.nodes[w]['sign']

def coin(p : float):
    return np.random.choice([True, False], p=[1 - p, p])

def invert(sign : int):
    return -1 * sign

def sign_partition(G : nx.graph, p_pos : float = 0.5):
    p_neg = 1 - p_pos
    random_signs = np.random.choice([-1, 1], G.number_of_edges(), p=[p_neg, p_pos])
    nx.set_edge_attributes(G, dict(zip(G.edges(), random_signs)), 'sign')
    return G