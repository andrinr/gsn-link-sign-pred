# External dependencies
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
# Local dependenvies
from simulation.BSCL import BSCL
from denoising import DenoisingModel

@hydra.main(version_base=None, config_path="conf", config_name="config")
def DDSNG(cfg : DictConfig) -> None:
    # Generate dataset
    bscl = BSCL(cfg.simulation.BSCL)
    bscl.run(cfg.simulation.n_simulations, cfg.simulation.n_nodes)

    #denoisingModel = DenoisingModel(cfg)

if __name__ == "__main__":
    DDSNG()