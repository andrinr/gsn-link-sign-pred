import os
import os.path as osp
import torch.nn.functional as F
from typing import Callable, Optional

import numpy as np
import torch
import itertools
import pandas as pd
from sklearn.preprocessing import LabelEncoder  

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_gz,
)

from torch_geometric.utils import coalesce

class WikiRFA(InMemoryDataset):
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

    url = 'https://snap.stanford.edu/data/{}'

    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 one_hot_signs: Optional[bool] = False):
        self.raw_name = 'wiki-RfA'
        self.names = ['meta.wikisigned-k2', 'out.wikisigned-k2', 'README.wikisigned-k2']
        self.one_hot_signs = one_hot_signs
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        return "wiki-RfA.txt"

    @property
    def processed_file_names(self) -> str:
        return 'wiki-RfA.pt'
        
    def download(self):
        path = download_url(self.url.format(self.raw_name + '.txt.gz'), self.raw_dir)
        extract_gz(path, self.raw_dir)

    def process(self):
        data = Data()
       # Read in file
        df_raw = pd.read_csv(self.raw_paths[0], sep='\t', header=None)
        df_raw = df_raw.to_numpy()

        u = []
        v = []
        signs = []
        for i in range(0, df_raw.shape[0], 7):
            src = df_raw[i, 0].split(":")
            tgt = df_raw[i+1, 0].split(":")
            vot = df_raw[i+2, 0].split(":")

            u.append(src[1])
            v.append(tgt[1])
            signs.append(vot[1])

            if i > 10000:
                break
                
        #le = LabelEncoder()
        #le.fit_transform(y_train
        u = np.array(u)
        v = np.array(v)
        signs = np.array(signs)

        """
        data.edge_index = torch.tensor(np.array(raw[:,:2].T), dtype=torch.long)
        data.edge_attr = torch.tensor(signs, dtype=torch.long)
        # convert to one-hot
        if self.one_hot_signs:
            # convert to 0 and 1
            data.edge_attr = torch.div(data.edge_attr + 1, 2, rounding_mode='trunc')
            data.edge_attr = F.one_hot(data.edge_attr, num_classes=2).float()
            
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])"""
