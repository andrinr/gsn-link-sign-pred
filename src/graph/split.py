from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import torch
import jax.numpy as jnp

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

    edge_index_train = jnp.array( data.edge_index)
    node_degrees = jnp.bincount(edge_index_train[0])
    print(jnp.sum(node_degrees == 0))

    # connected = False
    # i = 0
    # while not connected:
    # i += 1
    # print(f"Permuting graph {i} times")
    if treat_as_undirected:
        mask = data.edge_index[0] < data.edge_index[1]
        data.edge_index = data.edge_index[:, mask]
        data.edge_attr = data.edge_attr[mask]

    num_total = data.edge_attr.shape[0]
    num_train = int(train_percentage * num_total)
    num_test = num_total - num_train

    degs = torch.bincount(data.edge_index[0])
    

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

    # create new graph with train edges and check if it is connected
    print(data.edge_index.shape )
    edge_index_train = data.edge_index[:, train_mask]

    edge_index_train = jnp.array(edge_index_train)

    node_degrees = jnp.bincount(edge_index_train[0])
    print(jnp.sum(node_degrees == 0))
    connected = jnp.all(node_degrees > 0)
    
    return data, train_mask, test_mask