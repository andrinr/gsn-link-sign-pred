pos = train_dataset[0].x
    data = train_dataset[0]

    x = pos[:, 0]
    y = pos[:, 1]
    G = to_networkx(data).to_undirected()

    
    nx_pos = {i:(x[i-1], y[i-1]) for i in G.nodes()}
    
    edge_weights = (data.edge_attr + 1).tolist()

    # Map edge weights to different colors
    edge_colors = [plt.cm.jet(w) for w in edge_weights]

    # draw with networkx
    nx.draw_networkx_nodes(G, nx_pos)
    nx.draw_networkx_edges(G, nx_pos, edge_color=edge_colors)
    plt.show()

    plt.close('all')