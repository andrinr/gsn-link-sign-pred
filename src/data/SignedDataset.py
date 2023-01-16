import torch
from typing import Any, Callable, Dict, List, Optional, Union
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import LineGraph
from torch_geometric.explain import Explanation
from data.utils.samplers import even_uniform, even_exponential
from torch_geometric.datasets.graph_generator import GraphGenerator

class SignedDataset(InMemoryDataset):
    """
    Generates a synthetic signed dataset for evaluation of signed network algorithms
    """

    def __init__(self, 
        graph_generator :  Union[GraphGenerator, str],
        graph_generator_kwargs: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable] = None,
        num_graphs: Optional[int] = None):
        super().__init__(root=None, transform=transform)

        self.graph_generator = graph_generator(**graph_generator_kwargs)

        if num_graphs is None:
            num_graphs = 1

        data_list: List[Explanation] = []
        for i in range(num_graphs):
            data_list.append(self.get_graph())

        self.data, self.slices = self.collate(data_list)

    def get_graph(self) -> Explanation:
        data = self.graph_generator()
        y = data.edge_attr
        data.edge_attr = None
        #for transform in self.transforms:
        #    data = transform(data)
        
        data.y = y
        return data
