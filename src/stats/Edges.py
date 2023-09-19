from torch_geometric.data import Data
from torch_geometric.utils import degree
import numpy as np
import graph

class Edges:
    def __init__(self, data: Data, n_edges : int = 10000, mask = np.ndarray, seed : int = None):
        self.data = data
        self.n_edges = n_edges
        self.mask = mask
        self.seed = seed

    def __call__(self):
        if self.seed : np.random.seed(self.seed)

        self.n_balanced = 0
        self.n_unbalanced = 0
       
        indices = np.arange(self.data.num_edges)
        if self.mask is not None:
            indices = indices[self.mask]

        indices_sampled = np.random.choice(indices, self.n_edges, replace=False)

        self.edges = []

        for e1 in indices_sampled:
            edge = Edge(self.data, self.mask, e1)

            self.n_balanced += edge.n_balanced
            self.n_unbalanced += edge.n_unbalanced

            self.edges.append(edge)

        total = self.n_balanced + self.n_unbalanced
        self.p_balanced = self.n_balanced / total
    
    def compare(self, predictions):
        if self.edges is None:
            raise Exception("Call() first")
        
        n = len(self.edges)

        confusion_matrix = np.zeros((2, 2), dtype=int)
        part_of_balanced = np.zeros((2, 2, 10), dtype=int)
        part_of_unbalanced = np.zeros((2, 2, 10), dtype=int)

        for i in range(n):
            edge = self.edges[i]
            actual = int(edge.sign.item()/ 2 + 1)
            prediction = int(predictions[i].item() / 2 + 1)

            confusion_matrix[actual, prediction] += 1

            n_balanced = edge.n_balanced
            n_unbalanced = edge.n_unbalanced
            for triangle in edge.triangles:
                if triangle.sign == 1:
                    n_balanced += 1
                else:
                    n_unbalanced += 1

            # because of the way the triangles are counted, we have to divide by 2
            n_balanced = int(n_balanced / 2)
            n_unbalanced = int(n_unbalanced / 2)

            max = 10
            n_balanced = min(n_balanced, 9)
            n_unbalanced = min(n_unbalanced, 9)

            part_of_balanced[actual, prediction, n_balanced] += 1
            
            part_of_unbalanced[actual, prediction, n_unbalanced] += 1

        return confusion_matrix, part_of_balanced, part_of_unbalanced
            
class Edge:
    def __init__(self, data : Data, mask : np.ndarray, uv : int):

        self.index = uv
        self.triangles = []
        self.n_balanced = 0
        self.n_unbalanced = 0
        self.sign = data.edge_attr[uv]

        u, v = data.edge_index[:, uv]

        neighs_u = set(graph.get_neighbors(data, u).tolist())
        neighs_v = set(graph.get_neighbors(data, v).tolist())
    
        neighs_u = neighs_u - {u}
        neighs_v = neighs_v - {v}

        # find all neighbors of u that are also neighbors of v
        neighs_u_and_v = neighs_u.intersection(neighs_v)
        
        for w in neighs_u_and_v:

            uw = graph.get_edge_index(data, u, w) 
            if len(uw) > 1:
                uw = uw[0]
            vw = graph.get_edge_index(data, v, w)
            if len(vw) > 1:
                vw = vw[0]

            # ignore triangles with neutral edges
            if mask[uw] or mask[vw]:
                continue

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

        self.uv_sign = data.edge_attr[uv]
        self.vu_sign = data.edge_attr[vw]
        self.uw_sign = data.edge_attr[uw]

        self.sign = self.uv_sign * self.vu_sign * self.uw_sign