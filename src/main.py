# External dependencies
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
# Local dependenvies
from simulation.BSCL import BSCL
from utils.utils import even_uniform, even_exponential
from denoising import DenoisingModel

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    #denoisingModel = DenoisingModel(cfg)

if __name__ == "__main__":
    my_app()