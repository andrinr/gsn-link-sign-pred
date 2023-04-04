
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
        return super().forward(x, index, ptr, dim_size, dim)
    

class RochaThate(MessagePassing):
    def __init__(self,
        collect_attributes: bool = True):
        super().__init__(aggr='add')

    def forward(self, edge_index, index, sign):
        return self.propagate(edge_index, index=index, sign=sign)
    
    def message(self, index_i, index_j, sign):
        return torch.cat((index_i, index_j, sign), 1)

class SpringTransform(BaseTransform):
    # implement this
    # https://en.wikipedia.org/wiki/Rocha%E2%80%93Thatte_cycle_detection_algorithm

    def __init__(self,
        max_cycles: int = 8,
        collect_attributes: bool = True):
        super().__init__()

        self.max_cycles = max_cycles
        self.collect_attributes = collect_attributes

    def __call__(self, data) -> Data:
        pass
