from torch_geometric.data import Data
from torch_geometric.utils import degree
import numpy as np
import matplotlib.pyplot as plt
import torch

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

        print("degs[current]", degs[current])
        
        visited = set()
        visited.add(current)

        for i in range(self.distance):
            new_nodes = []
            for node in nodes:
                print("node", node)
                visited.add(node)
                neighbors = get_neighbors(self.data, node)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        new_nodes.append(neighbor)
                        all_nodes.append(neighbor)

            nodes = new_nodes

        num_nodes = len(all_nodes)

        # new figure
        plt.figure()
        plt.title("Section of graph")

        # plot nodes
        pos = torch.zeros((num_nodes, 2), device=self.data.edge_index.device)

        for i, node in enumerate(all_nodes):
            pos[i] = self.data.x[node, 0:2]

        pos = pos.cpu()

        plt.scatter(pos[:,0], pos[:,1], color="black")

        # plot edges

        self.data = self.data.cpu()

        for node in all_nodes:
            print("node", node)

            neighbors = get_neighbors(self.data, node)
            for neighbor in neighbors:
                if neighbor in nodes:
                    plt.plot(
                        [self.data.x[node,0], self.data.x[neighbor,0]], 
                        [self.data.x[node,1], self.data.x[neighbor,1]], color="black")

        plt.show()