import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
import graph

def gen_graph(n : int) -> Data:
    edge_index = torch.zeros((2, n), dtype=torch.long)
    edge_attr = torch.ones((n, 1))
    for i in range(n):
        edge_index[0, i] = i
        edge_index[1, i] = i + 1
        edge_attr[i] = 1

    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = 101

    transform = ToUndirected()
    data = transform(data)

    assert not data.is_directed()

    return data

def test_split():
    n = 8
    data = gen_graph(n)
    p_train = 0.6

    train_data, test_data = graph.train_test_split(
        data, p_train)
    assert train_data.num_edges == n * 2
    assert test_data.num_edges == n * 2

    print(train_data.edge_attr)
    print(train_data.edge_index)
    assert train_data.is_undirected()
    assert test_data.is_undirected()

    print(train_data.edge_attr.sum().item())
    print(test_data.edge_attr.sum().item())

    num_train = int(p_train * data.num_edges / 2)
    num_total = int(data.num_edges / 2)
    num_test = num_total - num_train

    assert train_data.edge_attr.sum().item() == num_train * 2
    assert test_data.edge_attr.sum().item() == num_test * 2
