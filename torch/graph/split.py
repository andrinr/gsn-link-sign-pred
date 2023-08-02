from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import torch

def train_test_split(
        data : Data, 
        train_percentage : float):
    """
    Sets the edge_attr of the test edges to 0. 

    Parameters
    ----------
    data : Data
        Un undirected signed graph.
    train_percentage : float
        The percentage of edges to use for training.
    test_percentage : float
        The percentage of edges to use for testing.
    """

    num_train = int(train_percentage * data.num_edges / 2)
    num_total = int(data.num_edges / 2)
    num_test = num_total - num_train

    assert data.is_undirected()
    assert data.edge_attr is not None

    mask = data.edge_index[0] < data.edge_index[1]
    data.edge_index = data.edge_index[:, mask]
    data.edge_attr = data.edge_attr[mask]

    perm = torch.randperm(num_total, device=data.edge_index.device)

    data.edge_index = data.edge_index[:, perm]
    data.edge_attr = data.edge_attr[perm]
    
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip([0])], dim=-1)
    data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
    
    train_edge_attr = data.edge_attr.clone()
    train_edge_attr[:num_test] = 0
    train_edge_attr[num_total:num_test+num_total] = 0

    test_edge_attr = data.edge_attr.clone()
    test_edge_attr[num_test:num_total] = 0
    test_edge_attr[num_total+num_test::] = 0
    
    training_data = Data(
        edge_index=data.edge_index,
        edge_attr=train_edge_attr,
        num_nodes=data.num_nodes)
    
    test_data = Data(
        edge_index=data.edge_index,
        edge_attr=test_edge_attr,
        num_nodes=data.num_nodes)
    
    return data, training_data, test_data