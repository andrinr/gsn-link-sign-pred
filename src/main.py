# External dependencies
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
import hydra
import pathlib
from omegaconf import DictConfig
# Local dependenvies
from simulation.BSCL_dataset import BSCLDataset
from denoising import DenoisingModel

@hydra.main(version_base=None, config_path="conf", config_name="config")
def DDSNG(cfg : DictConfig) -> None:
    # Generate dataset
    bscl = BSCLDataset(cfg=cfg.dataset, root=pathlib.Path(__file__).parent.resolve())

    # add noise 

    denoisingModel = DenoisingModel(cfg.model)

if __name__ == "__main__":
    DDSNG()