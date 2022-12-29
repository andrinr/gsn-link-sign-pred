import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def diffuse(signed_adj_mat : np.ndarray,  steps : int):
    # TODO: handle upper lower triangular matrix
    total = signed_adj_mat.shape[0] * signed_adj_mat.shape[1]
    negative_fraction = np.count_nonzero(signed_adj_mat == -1) / total
    positive_fraction = np.count_nonzero(signed_adj_mat == 1) / total
    neutral_likelihood = 1.0 - (negative_fraction + positive_fraction)

    change_likelihood = np.random.random(signed_adj_mat.shape)
    change_to = np.random.choice([-1, 0, 1], size=signed_adj_mat.shape, p=[negative_fraction, neutral_likelihood, positive_fraction])

    series = []
    series.append(signed_adj_mat)
    # the very last step should be a more ore less random matrix with similar distribution of positive and negative values
    for step in range(steps):
        p_change = 1.0 / steps * (step + 1)

        change_mask = change_likelihood < p_change
        
        diffused_mat = signed_adj_mat.copy()
        diffused_mat[change_mask] = change_to[change_mask]

        series.append(diffused_mat)

    return series