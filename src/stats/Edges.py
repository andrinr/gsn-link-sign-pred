from torch_geometric.data import Data
from torch_geometric.utils import degree
import numpy as np
import graph

class Edges:
    def __init__(self, data: Data):
        self.data = data

    def __call__(self, n_edges : int = 10000, seed: int = None):
        self.sample(n_edges, seed)
        self.stats()
    
    def sample(self, n_edges : int = 10000, seed: int = None):
        if seed : np.random.seed(seed)

        self.n_balanced = 0
        self.n_unbalanced = 0
       
        indices = np.random.choice(self.data.num_edges, n_edges, replace=False)

        self.edges = []

        for e1 in indices:
            edge = Edge(self.data, e1)

            self.n_balanced += edge.n_balanced
            self.n_unbalanced += edge.n_unbalanced

            self.edges.append(edge)

        total = self.n_balanced + self.n_unbalanced
        self.p_balanced = self.n_balanced / total
    
    def compare(self, predictions, test_mask):
        if self.edge_signs is None:
            raise Exception("Call generate() first")
        
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

class Edge:
    def __init__(self, data : Data, uv : int):

        self.index = uv
        self.triangles = []
        self.n_balanced = 0
        self.n_unbalanced = 0

        u, v = self.data.edge_index[:, uv]

        neighs_u = set(graph.get_neighbors(self.data, u).tolist())
        neighs_v = set(graph.get_neighbors(self.data, v).tolist())
    
        neighs_u = neighs_u - {u}
        neighs_v = neighs_v - {v}

        # find all neighbors of u that are also neighbors of v
        neighs_u_and_v = neighs_u.intersection(neighs_v)
        
        for w in neighs_u_and_v:

            uw = graph.get_edge_index(self.data, u, w) 
            if len(uw) > 1:
                uw = uw[0]
            vw = graph.get_edge_index(self.data, v, w)
            if len(vw) > 1:
                vw = vw[0]

            triangle = Triangle(data, uv, vw, uw)

            if triangle.sign == 1:
                self.n_balanced += 1
            else:
                self.n_unbalanced += 1

            self.triangles.append(triangle)

class Triangle:
    def __init__(self, data : Data, uv : int, vw : int, uw : int):
        self.uv = uv
        self.vw = vw
        self.uw = uw

        self.uv_sign = self.data.edge_attr[uv]
        self.vu_sign = self.data.edge_attr[vw]
        self.uw_sign = self.data.edge_attr[uw]

        self.sign = self.uv_sign * self.vu_sign * self.uw_sign