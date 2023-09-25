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

        degs = degree(self.data.edge_index[0], n)

        current = np.random.randint(0, n)
        nodes = [current]

        all_nodes = [current]

        while (degs[current] < 2):
            current = np.random.randint(0, n)
        
        visited = set()
        visited.add(current)

        for i in range(self.distance):

            print("i", i)

            new_nodes = []
            for node in nodes:
                visited.add(node)
                neighbors = get_neighbors(self.data, node)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        new_nodes.append(neighbor)
                        all_nodes.append(neighbor)

            nodes = new_nodes

        # new figure
        plt.figure()
        plt.title("Section of graph")

        # plot nodes
        xs = []
        ys = []

        self.data = self.data.to("cpu")

        for node in all_nodes:
            print(node)
            xs.append(self.data.x[node, 0])
            ys.append(self.data.x[node, 1])

        plt.scatter(xs, ys, color="black")

        # plot edges

        for node in all_nodes:
            neighbors = get_neighbors(self.data, node)
            for neighbor in neighbors:
                if neighbor in nodes:
                    plt.plot([node, neighbor], [node, neighbor], color="black")

        plt.show()




    
            

        