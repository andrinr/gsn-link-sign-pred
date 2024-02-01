from torch_geometric.data import Data
from torch_geometric.utils import degree
import numpy as np
import graph

class Triplets:
    def __init__(self, data: Data):
        self.data = data

    def __call__(self, n_triplets : int = 1000, seed: int = None):
        self.sample(n_triplets, seed)
        self.stats()
        return self

    def sample(self, n_triplets : int = 1000, seed: int = None):
        if seed : np.random.seed(seed)
        self.triplets = []
        n_edges = self.data.num_edges

        n_found = 0

        # repeat until correct number of triplets is found
        while n_found < n_triplets:
            e1 = np.random.randint(n_edges, size=1)[0]
            u, v = self.data.edge_index[:, e1]

            neighs_u = set(graph.get_neighbors(self.data, u).tolist())
            neighs_v = set(graph.get_neighbors(self.data, v).tolist())

            neighs_u = neighs_u - {v}
            neighs_v = neighs_v - {u}

            # find a neighbor of u that is also a neighbor of v
            neighs_u_and_v = neighs_u.intersection(neighs_v)

            if len(neighs_u_and_v) == 0:
                continue

            # pick a random neighbor of u that is also a neighbor of v
            w = np.random.choice(list(neighs_u_and_v))

            e2 = graph.get_edge_index(self.data, u, w)
            if len(e2) > 1:
                e2 = e2[0]
            e3 = graph.get_edge_index(self.data, v, w)
            if len(e3) > 1:
                e3 = e3[0]

            e2 = e2.item()
            e3 = e3.item()

            triangle = frozenset([e1, e2, e3])

            if triangle not in self.triplets:
                self.triplets.append(triangle)
                n_found += 1


    def stats(self):
        n_balanced = 0
        n_unbalanced = 0

        self.edge_signs = []

        print(self.data.edge_attr)

        for triplet in self.triplets:
            # access elements of triplet which is a frozenset
            triplet = list(triplet)
            if len(triplet) != 3:
                continue

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

        self.p_balanced = n_balanced / (n_balanced + n_unbalanced)

        return self
    
    def compare(self, predictions, test_mask):
        if self.edge_signs is None:
            raise Exception("Call() first")
        
        n = len(self.triplets)

        total_balanced = np.zeros(3)
        correct_balanced = np.zeros(3)

        total_unbalanced = np.zeros(3)
        correct_unbalanced = np.zeros(3)

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

            for j in range(3):
                if n_neutral == j + 1:
                    if actual == 1:
                        total_balanced[j] += 1
                        correct_balanced[j] += 1 if predicted == actual else 0
                    else:
                        total_unbalanced[j] += 1
                        correct_unbalanced[j] += 1 if predicted == actual else 0
        
        return total_balanced, correct_balanced, total_unbalanced, correct_unbalanced
    
        # change print string
        __str__ = __repr__ = lambda self: f"Triplets(n_triplets={self.n_triplets}, p_balanced={self.p_balanced})"