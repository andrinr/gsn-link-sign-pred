import helpers
from torch_geometric.data import Data
import torch

def gen_graph() -> Data:
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    
    data = Data(edge_index=edge_index)

    return data

def test_check_edge_exists():
    data = gen_graph()
    assert helpers.check_edge_exists(data, 0, 1)
    assert helpers.check_edge_exists(data, 1, 0)
    assert helpers.check_edge_exists(data, 1, 2)
    assert helpers.check_edge_exists(data, 2, 1)
    assert not helpers.check_edge_exists(data, 0, 2)
    assert not helpers.check_edge_exists(data, 2, 0)

def test_random_hop():
    data = gen_graph()
    assert helpers.random_hop(data, 0) in [1, 2]
    assert helpers.random_hop(data, 1) in [0, 2]
    assert helpers.random_hop(data, 2) in [0, 1]