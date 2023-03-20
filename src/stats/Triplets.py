from torch_geometric.data import Data
from torch_geometric.nn import functional as F
import numpy as np
import helpers

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
            u, v = self.data.edge_index[:, e1]

            neighs_u = set(helpers.get_neighbors(self.data, u))
            neighs_v = set(helpers.get_neighbors(self.data, v))

            neighs_u = neighs_u - {v}
            neighs_v = neighs_v - {u}

            # find a neighbor of u that is not a neighbor of v
            neighs_u_and_v = neighs_u.union(neighs_v)
            if len(neighs_u_and_v) == 0:
                continue

            w = np.random.choice(list(neighs_u_and_v))

            e2 = helpers.get_edge_index(self.data, u, w) 
            e3 = helpers.get_edge_index(self.data, v, w)

            print(e1, e2, e3)

            self.triplets.append((e1, e2, e3))

            n_found += 1

        return self

    def sign_stats(self):
        self.n_balanced = 0
        self.n_unbalanced = 0

        for triplet in self.triplets:
            uv = triplet[0]
            vu = triplet[1]
            uw = triplet[2]

            print(uv, vu, uw)

            uv_sign = self.data.edge_attr[uv]
            vu_sign = self.data.edge_attr[vu]
            uw_sign = self.data.edge_attr[uw]

            sign = uv_sign * vu_sign * uw_sign
            print(sign.item())
            if sign.item() == 1:
                self.n_balanced += 1
            else:
                self.n_unbalanced += 1

        return self
