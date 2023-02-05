import os
import os.path as osp
import torch.nn.functional as F
from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
)

from torch_geometric.utils import coalesce

class Slashdot(InMemoryDataset):
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
                 one_hot_signs: Optional[bool] = False):

        self.raw_name = 'download.tsv.slashdot-threads'
        self.names = ['meta.slashdot-threads', 'out.slashdot-threads', 'README.slashdot-threads']
        self.one_hot_signs = one_hot_signs
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return [osp.join('slashdot-threads', s) for s in self.names]

    @property
    def processed_file_names(self) -> str:
        return 'slashdot-threads.pt'
        
    def download(self):
        path = download_url(self.url.format(self.raw_name + '.tar.bz2'), self.raw_dir)
        extract_tar(path, self.raw_dir, "r:bz2")

    def process(self):
        data = Data()
        raw = np.genfromtxt(self.raw_paths[1], skip_header=2, dtype=np.int64)
        u = raw[:, 0]
        v = raw[:, 1]
        signs = raw[:, 2]

        data.edge_index = torch.tensor(np.array(raw[:,:2].T), dtype=torch.long)
        data.edge_attr = torch.tensor(signs, dtype=torch.long)
        # convert to one-hot
        if self.one_hot_signs:
            # convert to 0 and 1
            data.edge_attr = torch.div(data.edge_attr + 1, 2, rounding_mode='trunc')
            data.edge_attr = F.one_hot(data.edge_attr, num_classes=2).float()
            
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        #data.num_nodes = 51083
        #data.num_edges = 140778
        torch.save(self.collate([data]), self.processed_paths[0])
