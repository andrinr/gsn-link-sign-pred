import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn.models import Node2Vec
from model import MassSpring
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from model import OpinionLayer
from model import Opinion

class OpinionPE(BaseTransform):
    def __init__(
        self,
        device,
        epochs : int,
        pe_channels : int,
        hidden_channels : int,
    ):
        self.device = device
        self.epochs = epochs
        self.pe_channels = pe_channels
        self.hidden_channels = hidden_channels

        self.Opinion = Opinion(pe_channels, hidden_channels)

    def __call__(self, data: Data) -> Data:
        
        if self.device is not None:
            data = data.to(self.device)
            pos = torch.rand((data.num_nodes, self.pe_channels), device=self.device)
            hidden = torch.rand((data.num_nodes, self.hidden_channels), device=self.device)
        else:
            pos = torch.rand((data.num_nodes, self.pe_channels))
            hidden = torch.rand((data.num_nodes, self.hidden_channels))
            
        signs = data.edge_attr

        for i in tqdm(range(self.epochs)):
            pos = self.Opinion(pos, hidden, signs, data.edge_index)
            

        data.x = pos
        if self.device is None:
            return data
        else:
            return data.to('cpu')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'