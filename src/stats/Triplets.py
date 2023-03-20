from torch_geometric.data import Data
from torch_geometric.nn import functional as F
import numpy as np

class Triplets:
    def __init__(self, data: Data):
        self.data = data


    def sample(self, n_triplets : int = 1000, seed: int = 1):
        np.random.seed(seed)
        self.triplets = []
        n_edges = self.data.num_edges

        n_found = 0

         # sample 1000 triplets
        while n_found < n_triplets:
            e1 = np.random.randint(n_edges, size=1)[0]

            def two_hop_walk(data, u):
        neighbors = get_neighbors(data, u)
        if len(neighbors) == 0:
            return None
        v = np.random.choice(neighbors)
        neighbors = get_neighbors(data, v)
        if len(neighbors) == 0:
            return None
        w = np.random.choice(neighbors)
        return v, w


        return self

    def sign_stats(self):
        self.n_balanced = 0
        self.n_unbalanced = 0

        for triplet in self.triplets:
            uv = triplet[0]
            vu = triplet[1]
            uw = triplet[2]

            uv_sign = self.data.edge_attr[uv]
            vu_sign = self.data.edge_attr[vu]
            uw_sign = self.data.edge_attr[uw]

            sign = uv_sign * vu_sign * uw_sign
            print(sign)
            if sign == 1:
                self.n_balanced += 1
            else:
                self.n_unbalanced += 1

        return self
