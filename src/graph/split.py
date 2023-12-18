from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import torch

def permute_split(
    data : Data, 
    train_percentage : float,
    treat_as_undirected : bool = True
) -> tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    assert data.edge_attr is not None

    if treat_as_undirected:
        mask = data.edge_index[0] < data.edge_index[1]
        data.edge_index = data.edge_index[:, mask]
        data.edge_attr = data.edge_attr[mask]

    num_total = data.edge_attr.shape[0]
    num_train = int(train_percentage * num_total)
    num_test = num_total - num_train

    perm = torch.randperm(num_total, device=data.edge_index.device)

    data.edge_index = data.edge_index[:, perm]
    data.edge_attr = data.edge_attr[perm]
    
    if treat_as_undirected:
        data.edge_index = torch.cat([data.edge_index, data.edge_index.flip([0])], dim=-1)
        data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
    

    train_mask = torch.zeros(num_total * 2 if treat_as_undirected else num_total, dtype=torch.bool)
    train_mask[:num_train] = True
    train_mask[num_total:num_total + num_train] = True

    test_mask = torch.zeros(num_total * 2 if treat_as_undirected else num_total, dtype=torch.bool)
    test_mask[num_train:num_total] = True
    test_mask[num_total + num_train:] = True
    
    return data, train_mask, test_mask