from torch_geometric.data import Data
import torch
from torch_geometric.transforms import ToUndirected
import stats
import numpy as np

def gen_graph() ->  Data:
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        [1, 2, 0, 2, 0, 1, 3, 4, 5, 6, 2, 4, 2, 3, 2, 6, 2, 5]
        ], dtype=torch.long)
    
    edge_attr = torch.ones(edge_index.shape[1])
    edge_attr[0] = -1
    edge_attr[2] = -1
    
    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = 7

    return data

def test_triples():
    data = gen_graph()
    assert not data.is_directed()
    triplets = stats.Triplets(data).sample(n_triplets=300)
    triplets.sign_stats()
    print(triplets.p_balanced)
    assert np.abs(triplets.p_balanced - 0.66) < 0.05
