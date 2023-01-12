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
        super().__init__(root=None, transform = transform)

        self.graph_generator = GraphGenerator.resolve(
            graph_generator,
            **(graph_generator_kwargs or {}),
        )

        if num_graphs is None:
            num_graphs = 1

        data_list: List[Explanation] = []
        for i in range(num_graphs):
            data_list.append(self.get_graph())



    def get_graph(self, 
        num_infected_nodes: int,
                  max_path_length: int) -> Explanation:
        """
        Data is simulated instead of downloaded
        """
        self.raw_data_list = []

        for i in tqdm(range(self.cfg.n_simulations)):
            degrees = None
            if self.cfg.BSCL.sampling_method == "uniform":
                degrees = even_uniform(1, 20, self.cfg.n_nodes)
            elif self.cfg.BSCL.sampling_method == "exp":
                degrees = even_exponential(self.cfg.n_nodes, 5.0)
            else:
                raise Exception("Sampling method not implemented")

            regular_graph = generate_bscl_instance(
                degrees,
                self.cfg.BSCL.p_positive, 
                self.cfg.BSCL.p_close_triangle,
                self.cfg.BSCL.p_close_for_balance,
                self.cfg.BSCL.remove_self_loops)

            self.raw_data_list.append(pyg_graph)
