from torch_geometric.data import Data
from torch_geometric.utils import degree
import numpy as np
import matplotlib.pyplot as plt

from graph import get_neighbors

class Section:

    def __init__(self, distance : int, data : Data):
        self.distance = distance
        self.data = data

    def __call__(self):

        m = self.data.num_edges
        n = self.data.num_nodes

        current = np.random.randint(0, m)
        nodes = [current]

        degs = degree(self.data.edge_index[0], n)
        
        visited = set()
        visited.add(current)

        for i in range(self.distance):

            new_nodes = []
            for node in nodes:
                visited.add(node)
                neighbors = get_neighbors(node)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        new_nodes.append(neighbor)

            nodes = new_nodes

        # new figure
        plt.figure()
        plt.title("Section of graph")

        # plot nodes
        xs = []
        ys = []

        for node in nodes:
            xs.append(self.data.node_attr[node])
            ys.append(self.data.node_attr[node])

        plt.scatter(xs, ys, color="black")




    
            

        