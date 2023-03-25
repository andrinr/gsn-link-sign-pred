import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
import helpers

def gen_graph() -> Data:
    edge_index = torch.zeros((2, 100), dtype=torch.long)
    edge_attr = torch.ones((100, 1))
    for i in range(100):
        edge_index[0, i] = i
        edge_index[1, i] = i + 1
        edge_attr[i] = 1

    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = 101

    transform = ToUndirected()
    data = transform(data)

    return data

def test_split():
    data = gen_graph()
    train_data, test_data = helpers.train_test_split(
        data, 0.75, 0.25)
    assert train_data.num_edges == 100
    assert test_data.num_edges == 100
    assert train_data.edge_attr.sum().item() == 75
