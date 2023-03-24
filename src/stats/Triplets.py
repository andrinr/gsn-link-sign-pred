from torch_geometric.data import Data
from torch_geometric.nn import functional as F
import numpy as np
import helpers

class Triplets:
    def __init__(self, data: Data):
        self.data = data

    def sample(self, n_triplets : int = 1000, seed: int = None):
        if seed : np.random.seed(seed)
        self.triplets = []
        n_edges = self.data.num_edges

        n_found = 0

         # sample 1000 triplets
        while n_found < n_triplets:
            e1 = np.random.randint(n_edges, size=1)[0]
            u, v = self.data.edge_index[:, e1]

            neighs_u = set(helpers.get_neighbors(self.data, u).tolist())
            neighs_v = set(helpers.get_neighbors(self.data, v).tolist())

            neighs_u = neighs_u - {v}
            neighs_v = neighs_v - {u}

            # find a neighbor of u that is not a neighbor of v
            neighs_u_and_v = neighs_u.intersection(neighs_v)
            if len(neighs_u_and_v) == 0:
                continue

            w = np.random.choice(list(neighs_u_and_v))

            e2 = helpers.get_edge_index(self.data, u, w) 
            e3 = helpers.get_edge_index(self.data, v, w)

            self.triplets.append((e1, e2, e3))

            n_found += 1

        return self

    def generate(self):
        n_balanced = 0
        n_unbalanced = 0

        for triplet in self.triplets:
            uv = triplet[0]
            vu = triplet[1]
            uw = triplet[2]

            uv_sign = self.data.edge_attr[uv]
            vu_sign = self.data.edge_attr[vu]
            uw_sign = self.data.edge_attr[uw]

            sign = uv_sign * vu_sign * uw_sign
            if sign.item() == 1:
                n_balanced += 1
            else:
                n_unbalanced += 1

        n = len(self.triplets)

        self.p_balanced = n_balanced / n

        return self
