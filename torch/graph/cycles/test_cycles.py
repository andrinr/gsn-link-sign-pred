import graph.cycles.rocha_thate as rocha_thate
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, remove_self_loops
from torch_geometric.transforms import ToUndirected
from networkx.algorithms.cycles import simple_cycles
from graph.cycles import RochaThateCycles

def gen_graph() -> Data:
    n_nodes = 10
    n_edges = 15
    edge_index = torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)

    data = Data(edge_index=edge_index, num_nodes=n_nodes, num_edges=n_edges)

    transform = ToUndirected()
    data = transform(data)

    edge_attrs = torch.zeros((n_edges, 1), dtype=torch.long)
    
    data.edge_attr = edge_attrs

    return data

def test_find_cycles():
    length_bound = 4
    torch.manual_seed(1)
    data = gen_graph()
    G = to_networkx(data)
    print(f"NX graph: {G}")
    nx_cycles = sorted(simple_cycles(G, length_bound=length_bound))

    print(f"Found cycles: {nx_cycles}")

    transform = RochaThateCycles(max_cycles=length_bound, collect_attributes = False)

    data = transform(data)

    assert False