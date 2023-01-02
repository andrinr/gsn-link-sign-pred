import pandas as pd
import torch
from torch_geometric.data import Data

def load_graph(path):
    df = pd.read_csv('datasets/slashdot.csv', sep=',', header=0, index_col = 0)
    
    graph_array = df.to_numpy()

    edge_index = torch.tensor(graph_array[:,:2].T, dtype=torch.long)
    edge_attr = torch.tensor(graph_array[:,2], dtype=torch.float)

    data = Data(edge_index=edge_index, edge_attr=edge_attr)

    data.coalesce()

    return data