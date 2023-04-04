
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class SpringTransform(BaseTransform):
    # implement this
    # https://en.wikipedia.org/wiki/Rocha%E2%80%93Thatte_cycle_detection_algorithm

    