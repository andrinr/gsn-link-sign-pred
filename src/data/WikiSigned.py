import os
import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
    extract_zip,
)

from torch_geometric.utils import coalesce


class WikiSigned(InMemoryDataset):
    r"""The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

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

    def __init__(self, root: str, geom_gcn_preprocess: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = 'download.tsv.wikisigned-k2.tar.bz2'
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        # Returns, e.g., ['LINUX/train', 'LINUX/test']
        return self.name

    @property
    def processed_file_names(self) -> str:
        # Returns, e.g., ['LINUX_training.pt', 'LINUX_test.pt']
        return self.name
        
    def download(self):
        print('downloading')
        path = download_url(self.url.format(self.name), self.raw_dir)
        print(path)
        extract_tar(path, self.raw_dir)

    def process(self):

        pass
