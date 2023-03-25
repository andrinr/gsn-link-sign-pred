import graph
from torch_geometric.data import Data
import torch
import numpy as np

def gen_graph() -> Data:
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    
    data = Data(edge_index=edge_index)

    return data

def test_check_edge_exists():
    data = gen_graph()
    assert graph.check_edge_exists(data, 0, 1)
    assert graph.check_edge_exists(data, 1, 0)
    assert graph.check_edge_exists(data, 1, 2)
    assert graph.check_edge_exists(data, 2, 1)
    assert not graph.check_edge_exists(data, 0, 2)
    assert not graph.check_edge_exists(data, 2, 0)

def test_random_hop():
    data = gen_graph()
    assert graph.random_hop(data, 0) in [1, 2]
    assert graph.random_hop(data, 1) in [0, 2]
    assert graph.random_hop(data, 2) in [0, 1]

def test_get_neighbors():
    data = gen_graph()
    assert np.all(graph.get_neighbors(data, 0).numpy() == np.array([1]))
    assert np.all(graph.get_neighbors(data, 1).numpy() == np.array([0, 2]))
    assert np.all(graph.get_neighbors(data, 2).numpy() == np.array([1]))

def test_get_edge_index():
    data = gen_graph()
    assert graph.get_edge_index(data, 0, 1).item() == 0
    assert graph.get_edge_index(data, 1, 0).item() == 1
    assert graph.get_edge_index(data, 1, 2).item() == 2
    assert graph.get_edge_index(data, 2, 1).item() == 3
