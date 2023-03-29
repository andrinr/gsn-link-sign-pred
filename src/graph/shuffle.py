from torch_geometric.data import Data
import torch

def shuffle(data : Data):
    num_total = int(data.num_edges / 2)

    assert data.is_undirected()
    assert data.edge_attr is not None

    edge_index = data.edge_index[:, data.edge_index[0] <= data.edge_index[1]]
    edge_attr = data.edge_attr[data.edge_index[0] <= data.edge_index[1]]

    perm = torch.randperm(num_total, device=data.edge_index.device)

    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    
    edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=-1)
    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    return edge_index, edge_attr