import numpy as np
import torch

class Feature():
    """
    Feature class to store the feature size and hidden size of a feature.
    """
    def __init__(
        self,
        feature_size : int,
        hidden_size : int
    ):
        self.feature_size = feature_size
        self.hidden_size = hidden_size

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0