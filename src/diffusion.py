import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def diffuse(edge_features : np.ndarray,  steps : int):
    # TODO: handle upper lower triangular matrix
    shape = edge_features.shape
    total = shape[0] * shape[1]
    negative_fraction = np.count_nonzero(edge_features == -1) / total
    positive_fraction = np.count_nonzero(edge_features == 1) / total
    neutral_fraction = 1.0 - (negative_fraction + positive_fraction)

    change_likelihood = np.random.random(shape)
    change_to = np.random.choice([-1, 0, 1], size=shape, p=[negative_fraction, neutral_fraction, positive_fraction])

    series = []
    # the very last step should be a more ore less random matrix with similar distribution of positive and negative values
    for step in range(steps):
        p_change = 1.0 / steps * (step + 1)

        change_mask = change_likelihood < p_change
        
        diffused_mat = edge_features.copy()
        diffused_mat[change_mask] = change_to[change_mask]

        series.append(diffused_mat)

    return series