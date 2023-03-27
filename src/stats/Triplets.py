from torch_geometric.data import Data
from torch_geometric.nn import functional as F
import numpy as np
import graph

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

            neighs_u = set(graph.get_neighbors(self.data, u).tolist())
            neighs_v = set(graph.get_neighbors(self.data, v).tolist())

            neighs_u = neighs_u - {v}
            neighs_v = neighs_v - {u}

            # find a neighbor of u that is not a neighbor of v
            neighs_u_and_v = neighs_u.intersection(neighs_v)
            if len(neighs_u_and_v) == 0:
                continue

            w = np.random.choice(list(neighs_u_and_v))

            e2 = graph.get_edge_index(self.data, u, w) 
            if len(e2) > 1:
                e2 = e2[0]
            e3 = graph.get_edge_index(self.data, v, w)
            if len(e3) > 1:
                e3 = e3[0]

            self.triplets.append((e1, e2, e3))

            n_found += 1

        return self

    def stats(self):
        n_balanced = 0
        n_unbalanced = 0

        self.edge_signs = []

        print(self.data.edge_attr)

        for triplet in self.triplets:
            uv = triplet[0]
            vu = triplet[1]
            uw = triplet[2]

            uv_sign = self.data.edge_attr[uv]
            vu_sign = self.data.edge_attr[vu]
            uw_sign = self.data.edge_attr[uw]

            sign = uv_sign * vu_sign * uw_sign

            self.edge_signs.append(sign)

            if sign == 1:
                n_balanced += 1
            else:
                n_unbalanced += 1

        n = len(self.triplets)

        self.p_balanced = n_balanced / n

        return self
    
    def compare(self, predictions, test_mask):
        if self.edge_signs is None:
            raise Exception("Call generate() first")
        
        n = len(self.triplets)


        n_single_balanced = 0
        n_single_unbalanced = 0

        n_double_balanced = 0
        n_double_unbalanced = 0

        n_single_balanced_correct = 0
        n_single_unbalanced_correct = 0

        n_double_balanced_correct = 0
        n_double_unbalanced_correct = 0

        for i in range(n):
            e1, e2, e3 = self.triplets[i]

            t1 = test_mask[e1]
            t2 = test_mask[e2]
            t3 = test_mask[e3]

            s1 = predictions[e1] if t1 else self.data.edge_attr[e1]
            s2 = predictions[e2] if t2 else self.data.edge_attr[e2]
            s3 = predictions[e3] if t3 else self.data.edge_attr[e3]

            predicted = s1 * s2 * s3
            actual = self.edge_signs[i]

            n_neutral = int(t1.item()) + int(t2.item()) + int(t3.item())

            if n_neutral == 0:
                continue

            if n_neutral == 1:
                n_single_balanced_correct += 1 if predicted == actual and actual == 1 else 0
                n_single_unbalanced_correct += 1 if predicted == actual and actual == -1 else 0

                n_single_balanced += 1 if actual == 1 else 0
                n_single_unbalanced += 1 if actual == -1 else 0

            if n_neutral == 2:
                n_double_balanced_correct += 1 if predicted == actual and actual == 1 else 0
                n_double_unbalanced_correct += 1 if predicted == actual and actual == -1 else 0

                n_double_balanced += 1 if actual == 1 else 0
                n_double_unbalanced += 1 if actual == -1 else 0

        return n_single_balanced_correct / n_single_balanced, \
            n_single_unbalanced_correct / n_single_unbalanced, \
            n_double_balanced_correct / n_double_balanced, \
            n_double_unbalanced_correct / n_double_unbalanced