import numpy as np

def random_edge_features(N : int, positive_fraction : float = 0.1, negative_fraction : float = 0.01):
    neutral_fraction = 1.0 - (negative_fraction + positive_fraction)
    return np.random.choice(
        [-1, 0, 1], 
        size=(N,N), 
        p=[negative_fraction, neutral_fraction, positive_fraction]
    )
