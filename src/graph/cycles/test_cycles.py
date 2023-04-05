import graph.cycles.rocha_thate as rocha_thate
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, remove_self_loops
from torch_geometric.transforms import ToUndirected
from networkx.algorithms.cycles import simple_cycles
import graph.cycles as cycles

def gen_graph() -> Data:
    n_nodes = 10
    n_edges = 15
    edge_index = torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)
    data = Data(edge_index=edge_index, num_nodes=n_nodes)

    transform = ToUndirected()
    data = transform(data)

    return data

def test_find_cycles():
    length_bound = 4
    data = gen_graph()
    G = to_networkx(data)
    nx_cycles = sorted(simple_cycles(G, length_bound=length_bound))

    transform = cycles.RochaThateCycles(max_cycles=length_bound, collect_attributes = False)
    data = transform(data)

    assert False