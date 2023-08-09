
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from graph import get_cycles

class CycleTransform(BaseTransform):

    def __init__(self, max_degree : int):
        self.max_degree = max_degree

    def __call__(self, data: Data) -> Data:
        
        cycles = get_cycles(data, self.max_degree)
        
        node_attr = Data(cycles)

        # add node attributes to data
        data.node_attr = node_attr

        return data

