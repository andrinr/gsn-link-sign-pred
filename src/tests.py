import networkx as nx 
import matplotlib.pyplot as plt
from simulation.BSCL import BSCL
from utils.utils import even_uniform, even_exponential

G = BSCL(even_exponential(5, 100))
edges = G.edges()
pos = nx.spring_layout(G, seed=63)
colors = [G[u][v]['sign'] for u, v in edges]

nx.draw(G, pos, edge_color=colors, node_size=10)
plt.show()