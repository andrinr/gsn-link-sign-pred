import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def visualize(data):
    G = to_networkx(data, to_undirected=True)

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=100)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.axis('off')
    plt.show()