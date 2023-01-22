import os
import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_bz2,
    extract_tar,
)

from torch_geometric.utils import coalesce

class WikiSigned(InMemoryDataset):
    r"""
    This undirected signed network contains interpreted interactions between the users of the English Wikipedia that have edited pages about politics. 
    Each interaction, such as text editing, reverts, restores and votes are given a positive or negative value. 
    The result can be interpreted as a web of trust. An edge represents an interaction and a node represents a user.
    Note that a user may revert his own edits, and thus this network contains negatively weighted loops.  
    The dataset is based on a set of 563 articles from the politics domain of the English Wikipedia. 

    Silviu Maniu, Talel Abdessalem, Bogdan Cautis. Casting a web of trust over Wikipedia: An interaction-based approach. ResearchGate. Published March 28, 2011. Accessed January 19, 2023. 
    https://www.researchgate.net/publication/221023171_Casting_a_web_of_trust_over_Wikipedia_An_interaction-based_approach

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    url = 'http://konect.cc/files/{}'

    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 map_to_zero_one: Optional[bool] = False):
        self.raw_name = 'download.tsv.wikisigned-k2'
        self.names = ['meta.wikisigned-k2', 'out.wikisigned-k2', 'README.wikisigned-k2']
        self.map_to_zero_one = map_to_zero_one
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        # Returns, e.g., ['LINUX/train', 'LINUX/test']
        return [osp.join('wikisigned-k2', s) for s in self.names]

    @property
    def processed_file_names(self) -> str:
        # Returns, e.g., ['LINUX_training.pt', 'LINUX_test.pt']
        return 'wikisigned-k2.pt'
        
    def download(self):
        path = download_url(self.url.format(self.raw_name + '.tar.bz2'), self.raw_dir)
        extract_tar(path, self.raw_dir, "r:bz2")

    def process(self):
        data = Data()
        raw = np.genfromtxt(self.raw_paths[1], skip_header=1, dtype=np.int64)
        u = raw[:, 0]
        v = raw[:, 1]
        signs = raw[:, 2]

        data.edge_index = torch.tensor(np.array(raw[:,:2].T), dtype=torch.long)
        if self.map_to_zero_one:
            signs = (signs + 1) / 2
        data.edge_attr = torch.tensor(signs, dtype=torch.float)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
