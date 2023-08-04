
import torch
from torch import Tensor
from typing import Optional
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Aggregation

class RochaThatteAggregation(Aggregation):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        
        print(self.reduce(x, index, ptr, dim_size, dim, reduce='sum'))
        print("x", x)
        return x
    
class RochaThatteIteration(MessagePassing):
    def __init__(self,
        collect_attributes: bool = True):
        super().__init__(aggr=RochaThatteAggregation())

    def forward(self, edge_index, indices, sign):
        return self.propagate(edge_index, indices=indices, sign=sign)
    
    def message(self, index_i, index_j, sign):
        return index_j

class RochaThatteCycles():
    # implement this
    # https://en.wikipedia.org/wiki/Rocha%E2%80%93Thatte_cycle_detection_algorithm

    def __init__(self,
        max_cycles: int = 8,
        dimensions: int = 2,
        collect_attributes: bool = True):
        super().__init__()

        self.max_cycles = max_cycles
        self.dimensions = dimensions
        self.collect_attributes = collect_attributes

        self.iteration = RochaThatteIteration(collect_attributes = collect_attributes)

    def __call__(self, data) -> Data:
        ids = torch.arange(data.num_nodes, device=data.edge_index.device)
        indices = torch.zeros(data.num_nodes, device=data.edge_index.device)
        
        cycles = torch.zeros(
            (data.num_nodes, self.max_cycles), 
            device=data.edge_index.device)

        for k in range(self.max_cycles):
            print("Iteration", k)
            pos = self.iteration(
                edge_index = data.edge_index, 
                indices = indices, sign = data.edge_attr)
            
            # get all zero position indices
            k_cycle = torch.where(torch.any(indices == ids, dim=1))
            cycles[k_cycle, :] = k_cycle

        print("cycles", cycles)

        return data

def RochaThatte(data : Data, max_cycles: int = 8) -> torch.Tensor:
    assert data.is_undirected()
    
    # Convert to line graph, signs are new node attributes
    to_line_graph = LineGraph(force_directed=True)
    line_graph = to_line_graph(data)

    # Run Rocha-Thatte algorithm


