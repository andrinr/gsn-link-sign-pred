from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import torch

def train_test_split(
        data : Data, 
        train_percentage : float, 
        test_percentage : float):

    num_train = int(train_percentage * data.num_edges)
    num_test = int(test_percentage * data.num_edges)

    assert num_train + num_test == data.num_edges

    perm = torch.randperm(data.num_edges, device=data.edge_index.device)
    perm = perm[data.edge_index[0] <= data.edge_index[1]]
    edge_index = data.edge_index[:, perm]

    train_edges = perm[:num_train]
    test_edges = perm[num_train:num_train + num_test]

    train_edge_attr = data.edge_attr.clone()
    train_edge_attr[test_edges] = 0

    test_edge_attr = data.edge_attr.clone()
    test_edge_attr[train_edges] = 0

    train_data = Data(
        edge_index=edge_index, 
        num_nodes=data.num_nodes,
        edge_attr=train_edge_attr)
    
    test_data = Data(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        edge_attr=test_edge_attr)
    
    return train_data, test_data
