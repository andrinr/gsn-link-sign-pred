import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from simulation import SpringState, predict, SpringParams
from graph import SignedGraph
    
def plot_embedding(spring_state : SpringState, spring_params : SpringParams, graph : SignedGraph, axis : plt.Axes):
        # plot the embeddings
    embeddings = spring_state.position
    predicted_sign = predict(spring_state, spring_params, graph)

    axis.scatter(embeddings[:, 0], embeddings[:, 1])

    # add edges to plot
    for i in range(graph.edge_index.shape[1]):
        if graph.test_mask[i] == 1:
           
           axis.plot(
                [embeddings[graph.edge_index[0][i]][0], embeddings[graph.edge_index[1][i]][0]], 
                [embeddings[graph.edge_index[0][i]][1], embeddings[graph.edge_index[1][i]][1]], 
                color= 'blue' if graph.sign[i] == 1 else 'red',
                linestyle='dotted',
                alpha=1.0)
        
        else:
            axis.plot(
                [embeddings[graph.edge_index[0][i]][0], embeddings[graph.edge_index[1][i]][0]], 
                [embeddings[graph.edge_index[0][i]][1], embeddings[graph.edge_index[1][i]][1]], 
                color= 'blue' if graph.sign[i] == 1 else 'red',
                alpha=1.0)

    # add legend for edges
    axis.plot([], [], color='blue', label='positive')
    axis.plot([], [], color='red', label='negative')
    axis.plot([], [], color='black', linestyle='dashed', label='test set')
    axis.plot([], [], color='black', label='train & val set')
    
    axis.legend()
