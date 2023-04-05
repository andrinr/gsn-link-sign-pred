
import torch
from torch import Tensor
from typing import Optional
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import Aggregation

class RochaThateAggregation(Aggregation):
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        
        print("x", x)
        return x
    
class RochaThateIteration(MessagePassing):
    def __init__(self,
        collect_attributes: bool = True):
        super().__init__(aggr=RochaThateAggregation())

    def forward(self, edge_index, position, sign):
        return self.propagate(edge_index, position=position, sign=sign)
    
    def message(self, position_i, position_j, sign):
        vector = position_j - position_i
        return position_j + vector

class RochaThateCycles(BaseTransform):
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

        self.iteration = RochaThateIteration(collect_attributes = collect_attributes)

    def __call__(self, data) -> Data:
        pos = torch.rand(
            (data.num_nodes, self.dimensions), 
            device=data.edge_index.device) * 2.0 - 1.0

        pos = pos.unsqueeze(1)

        print(pos.shape)
        print(data.edge_index)
        print(data.num_nodes)

        for i in range(self.max_cycles):
            pos = self.iteration(
                edge_index = data.edge_index, 
                position = pos, sign = data.edge_attr)
            
            print(pos.shape)

        pass
