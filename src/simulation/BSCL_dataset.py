import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from simulation.BSCL_simulation import generate_bscl_instance
from utils.samplers import even_uniform, even_exponential

class BSCLDataset(InMemoryDataset):
    def __init__(self, cfg, root, transform=None, pre_transform=None, pre_filter=None):
        self.cfg = cfg
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['BSCL']

    @property
    def processed_file_names(self):
        return ['BSCL_processed.pt']

    # We just simulate instead of download the data
    def download(self):
        self.raw_data_list = []

        for i in tqdm(range(self.cfg.n_simulations)):
            degrees = None
            if self.cfg.BSCL.sampling_method == "uniform":
                degrees = even_uniform(1, 20, self.cfg.n_nodes)
            elif self.cfg.BSCL.sampling_method == "exp":
                degrees = even_exponential(self.cfg.n_nodes, 5.0)
            else:
                raise Exception("Sampling method not implemented")

            pyg_data = generate_bscl_instance(
                degrees,
                self.cfg.BSCL.p_positive, 
                self.cfg.BSCL.p_close_triangle,
                self.cfg.BSCL.p_close_for_balance,
                self.cfg.BSCL.remove_self_loops)

            self.raw_data_list.append(pyg_data)
    
    def process(self):
        # Read data into huge `Data` list.
        data_list = self.raw_data_list
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        