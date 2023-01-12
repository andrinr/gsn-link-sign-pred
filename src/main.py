# External dependencies
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
import hydra
import pathlib
from omegaconf import DictConfig
# Local dependenvies
from data.BSCLGraph import BSCLGraph
from data.SignedDataset import SignedDataset
from data.utils.samplers import even_exponential
from denoising import DenoisingModel
import pytorch_geometric.transforms as T

@hydra.main(version_base=None, config_path="conf", config_name="config")
def DDSNG(cfg : DictConfig) -> None:

    BSCL_graph_kwargs = {
        "degree_generator": even_exponential,
        "degree_generator_kwargs": {"size": cfg.dataset.n_nodes, "scale": 5.0},
        "p_positive_sign": cfg.dataset.BSCL.p_positive,
        "p_close_triangle": cfg.dataset.BSCL.p_close_triangle,
        "p_close_for_balance": cfg.dataset.BSCL.p_close_for_balance,
        "remove_self_loops": cfg.dataset.BSCL.remove_self_loops
    }

    transform = T.Compose([T.LineGraph])

    dataset = SignedDataset(
        graph_generator=BSCLGraph,
        graph_generator_kwargs=BSCL_graph_kwargs,
        transform=transform,
        num_graphs=cfg.dataset.n_graphs)

    denoisingModel = DenoisingModel(cfg.model)

if __name__ == "__main__":
    DDSNG()