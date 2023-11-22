import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from simulation import SpringState, predict, SpringParams
from graph import SignedGraph
import simulation as sim
from graph import to_SignedGraph
import jax.numpy as jnp
import jax.random as random
import jax
import numpy as np
from sklearn.decomposition import PCA
from torch_geometric.data import Data
    
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

def params_plot(
        dataset : Data, 
        key : jax.random.PRNGKey, 
        spring_params : SpringParams, 
        init_range : float,
        dim : int,
        iterations : int,
        dt : float,
        damping : float):
    
    shots = 32

    auc_avgs = []
    f1_binary_avgs = []
    f1_micro_avgs = []
    f1_macro_avgs = []

    auc_vars = []
    f1_binary_vars = []
    f1_micro_vars = []
    f1_macro_vars = []

    dims = [2, 4, 8, 16, 32, 64, 128]
    for dim in dims:
        shot_metrics = []
        key_shots = random.split(key, shots)
        for shot in range(shots):

            graph = to_SignedGraph(dataset)
            print(graph.sign)

            print(f"shot: {shot}")

            # initialize spring state
            spring_state = sim.init_spring_state(
                rng=key_shots[shot],
                n=graph.num_nodes,
                m=graph.num_edges,
                range=init_range,
                embedding_dim=dim
            )

            # training_signs = graphsigns.copy()
            # training_signs = training_signs.at[train_mask].set(0)

            simulation_params_test = sim.SimulationParams(
                iterations=iterations,
                dt=dt,
                damping=damping,
                message_passing_iterations=0)

            training_signs = graph.sign.copy()
            training_signs = jnp.where(graph.train_mask, training_signs, 0)
            training_graph = graph._replace(sign=training_signs)

            spring_state = sim.simulate(
                simulation_params=simulation_params_test,
                spring_state=spring_state, 
                spring_params=spring_params,
                nn_force=False,
                nn_force_params={},
                graph=training_graph)

            metrics = sim.evaluate(
                spring_state,
                graph.edge_index,
                graph.sign,
                graph.train_mask,
                graph.test_mask)
            
            shot_metrics.append(metrics)

        auc_avgs.append(np.mean([metrics.auc for metrics in shot_metrics]))
        f1_binary_avgs.append(np.mean([metrics.f1_binary for metrics in shot_metrics]))
        f1_micro_avgs.append(np.mean([metrics.f1_micro for metrics in shot_metrics]))
        f1_macro_avgs.append(np.mean([metrics.f1_macro for metrics in shot_metrics]))

        auc_vars.append(np.var([metrics.auc for metrics in shot_metrics]))
        f1_binary_vars.append(np.var([metrics.f1_binary for metrics in shot_metrics]))
        f1_micro_vars.append(np.var([metrics.f1_micro for metrics in shot_metrics]))
        f1_macro_vars.append(np.var([metrics.f1_macro for metrics in shot_metrics]))

    # new plot
    fig, ax = plt.subplots(1, 1)
    
    ax.plot(dims, auc_avgs)
    ax.plot(dims, f1_binary_avgs)
    ax.plot(dims, f1_micro_avgs)
    ax.plot(dims, f1_macro_avgs)
    ax.legend(['AUC', 'F1 binary', 'F1 micro', 'F1 macro'])
    ax.set_title('Measures')
    ax.set_xlabel('Embedding dimension')
    ax.set_xscale('log')

    ax.set_ylabel('Average measure over all shots')
    plt.show()

    
    iterations = [32, 64, 128, 256, 512, 1024, 2048]
    for iteration in iterations:
        shot_metrics = []
        key_shots = random.split(key, shots)
        for shot in range(shots):

            graph = to_SignedGraph(dataset)
            print(graph.sign)

            print(f"shot: {shot}")

            # initialize spring state
            spring_state = sim.init_spring_state(
                rng=key_shots[shot],
                n=graph.num_nodes,
                m=graph.num_edges,
                range=init_range,
                embedding_dim=64
            )

            # training_signs = graphsigns.copy()
            # training_signs = training_signs.at[train_mask].set(0)

            simulation_params_test = sim.SimulationParams(
                iterations=iteration,
                dt=dt,
                damping=damping,
                message_passing_iterations=0)

            training_signs = graph.sign.copy()
            training_signs = jnp.where(graph.train_mask, training_signs, 0)
            training_graph = graph._replace(sign=training_signs)

            spring_state = sim.simulate(
                simulation_params=simulation_params_test,
                spring_state=spring_state, 
                spring_params=spring_params,
                nn_force=False,
                nn_force_params={},
                graph=training_graph)

            metrics = sim.evaluate(
                spring_state,
                graph.edge_index,
                graph.sign,
                graph.train_mask,
                graph.test_mask)
            
            shot_metrics.append(metrics)

        auc_avgs.append(np.mean([metrics.auc for metrics in shot_metrics]))
        f1_binary_avgs.append(np.mean([metrics.f1_binary for metrics in shot_metrics]))
        f1_micro_avgs.append(np.mean([metrics.f1_micro for metrics in shot_metrics]))
        f1_macro_avgs.append(np.mean([metrics.f1_macro for metrics in shot_metrics]))
        
    # new plot
    fig, ax = plt.subplots(1, 1)
    
    ax.plot(iterations, auc_avgs)
    ax.plot(iterations, f1_binary_avgs)
    ax.plot(iterations, f1_micro_avgs)
    ax.plot(iterations, f1_macro_avgs)

    ax.legend(['AUC', 'F1 binary', 'F1 micro', 'F1 macro'])
    ax.set_title('Measures')
    ax.set_xlabel('Iterations')
    # set x axis to 2^x
    ax.set_xscale('log')
    ax.set_ylabel('Average measure over all shots')
    plt.show()
    
    shots = 32

    auc_avgs = []
    f1_binary_avgs = []
    f1_micro_avgs = []
    f1_macro_avgs = []

    auc_vars = []
    f1_binary_vars = []
    f1_micro_vars = []
    f1_macro_vars = []
    
    dampings = [0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]
    for damping in dampings:
        shot_metrics = []
        key_shots = random.split(key, 32)
        for shot in range(32):

            graph = to_SignedGraph(dataset)
            print(graph.sign)

            print(f"shot: {shot}")

            # initialize spring state
            spring_state = sim.init_spring_state(
                rng=key_shots[shot],
                n=graph.num_nodes,
                m=graph.num_edges,
                range=range,
                embedding_dim=64
            )

            # training_signs = graphsigns.copy()
            # training_signs = training_signs.at[train_mask].set(0)

            simulation_params_test = sim.SimulationParams(
                iterations=iterations,
                dt=dt,
                damping=damping,
                message_passing_iterations=0)

            training_signs = graph.sign.copy()
            training_signs = jnp.where(graph.train_mask, training_signs, 0)
            training_graph = graph._replace(sign=training_signs)

            spring_state = sim.simulate(
                simulation_params=simulation_params_test,
                spring_state=spring_state, 
                spring_params=spring_params,
                nn_force=Falses,
                nn_force_params={},
                graph=training_graph)

            metrics = sim.evaluate(
                spring_state,
                graph.edge_index,
                graph.sign,
                graph.train_mask,
                graph.test_mask)
            
            shot_metrics.append(metrics)

        auc_avgs.append(np.mean([metrics.auc for metrics in shot_metrics]))
        f1_binary_avgs.append(np.mean([metrics.f1_binary for metrics in shot_metrics]))
        f1_micro_avgs.append(np.mean([metrics.f1_micro for metrics in shot_metrics]))
        f1_macro_avgs.append(np.mean([metrics.f1_macro for metrics in shot_metrics]))

        auc_vars.append(np.var([metrics.auc for metrics in shot_metrics]))
        f1_binary_vars.append(np.var([metrics.f1_binary for metrics in shot_metrics]))
        f1_micro_vars.append(np.var([metrics.f1_micro for metrics in shot_metrics]))
        f1_macro_vars.append(np.var([metrics.f1_macro for metrics in shot_metrics]))

    # new plot
    fig, ax = plt.subplots(1, 1)
    ax.plot(dampings, auc_avgs)
    ax.plot(dampings, f1_binary_avgs)
    ax.plot(dampings, f1_micro_avgs)
    ax.plot(dampings, f1_macro_avgs)
    ax.legend(['AUC', 'F1 binary', 'F1 micro', 'F1 macro'])
    ax.set_title('Measures')
    ax.set_xlabel('Damping')
    ax.set_xscale('log')
    ax.set_ylabel('Average measure over all shots')
    plt.show()

    
def selected_wrong_classification(
        dataset : Data, 
        key : jax.random.PRNGKey, 
        spring_params : SpringParams, 
        init_range : float,
        dim : int,
        iterations : int,
        dt : float,
        damping : float):
    
    shots = 15
    graph = to_SignedGraph(dataset)
    shot_metrics = []
    key_shots = random.split(key, shots)    
    false_edges = jnp.zeros(graph.num_edges)

    for shot in range(shots):
        # print(graph.sign)
        graph = to_SignedGraph(dataset)

        print(f"shot: {shot}")

        # initialize spring state
        spring_state = sim.init_spring_state(
            rng=key_shots[shot],
            n=graph.num_nodes,
            m=graph.num_edges,
            range=init_range,
            embedding_dim=dim
        )

        # training_signs = graphsigns.copy()
        # training_signs = training_signs.at[train_mask].set(0)

        simulation_params_test = sim.SimulationParams(
            iterations=iterations,
            dt=dt,
            damping=damping,
            message_passing_iterations=0)

        training_signs = graph.sign.copy()
        training_signs = jnp.where(graph.train_mask, training_signs, 0)
        training_graph = graph._replace(sign=training_signs)

        spring_state = sim.simulate(
            simulation_params=simulation_params_test,
            spring_state=spring_state, 
            spring_params=spring_params,
            nn_force=False,
            nn_force_params={},
            graph=training_graph)

        metrics, y_pred = sim.evaluate(
            spring_state,
            graph.edge_index,
            graph.sign,
            graph.train_mask,
            graph.test_mask)

        false_negatives = jnp.logical_and(y_pred == -1, graph.sign.at[graph.test_mask].get() == 1)
        false_positives = jnp.logical_and(y_pred == 1, graph.sign.at[graph.test_mask].get() == -1)
        
        print(jnp.sum(false_negatives))
        print(jnp.sum(false_positives))

        false_edges = false_edges.at[graph.test_mask].add(false_negatives)
        false_edges = false_edges.at[graph.test_mask].add(false_positives)
        
        shot_metrics.append(metrics)

    # look at top 6 false negatives edges and their surrounding local graph structure
    false_edges = jnp.argsort(false_edges)[::-1]
    print(false_edges)

    # transform position to numpy array
    embeddings = np.array(spring_state.position)
    edge_index = np.array(graph.edge_index)
    signs = np.array(graph.sign)
    
    fig, axes = plt.subplots(2, 3)

    depth = 1
    found = 0
    i = 0
    visited_edges = {}
    while found < 6:

        edge = false_edges[i]
        i += 1

        nodes = edge_index[:, edge]
        nodes = np.unique(nodes)

        edge_node_a = nodes[0]
        edge_node_b = nodes[1]

        deg_node_a = np.sum(edge_index[0] == edge_node_a)
        deg_node_b = np.sum(edge_index[0] == edge_node_b)

        if deg_node_a + deg_node_b > 20 or \
            deg_node_a < 2 or \
            deg_node_b < 2 or \
            (edge_node_a, edge_node_b) in visited_edges or \
            (edge_node_b, edge_node_a) in visited_edges:
            continue

        visited_edges[(edge_node_a, edge_node_b)] = True
        visited_edges[(edge_node_b, edge_node_a)] = True

        for d in range(depth):
            for node in nodes:
                nodes = np.unique(np.concatenate((nodes, edge_index[0, edge_index[1] == node])))

        # get embeddings of nodes
        node_embeddings = embeddings[nodes]

        # reduce dimensionality of embeddings
        pca = PCA(n_components=2)
        node_embeddings = pca.fit_transform(node_embeddings)

        # plot embeddings
        axes[found // 3, found % 3].scatter(node_embeddings[:, 0], node_embeddings[:, 1])

        # plot edges
        for k in range(len(nodes)):
            node_a = nodes[k]
            neigbours = edge_index[0, edge_index[1] == node_a]

            for j in range(k + 1, len(nodes)):
                node_b = nodes[j]
                
                # check if neighbours array contains node_b
                exists = False
                for l in range(len(neigbours)):
                    if neigbours[l] == node_b:
                        exists = True
                        break

                if not exists:
                    continue

                # get id of edge between node_a and node_b
                edge_index_a = edge_index[0] == node_a
                edge_index_b = edge_index[1] == node_b
                edge_index_ab = np.logical_and(edge_index_a, edge_index_b)
                edge_index_ab = np.where(edge_index_ab)[0]

                axes[found // 3, found % 3].plot(
                    [node_embeddings[k, 0], node_embeddings[j, 0]],
                    [node_embeddings[k, 1], node_embeddings[j, 1]],
                    color='green' if signs[edge_index_ab] == 1 else 'red',
                    linestyle='dashed' if node_a == edge_node_a and node_b == edge_node_b else 'solid',
                    linewidth=3 if node_a == edge_node_a and node_b == edge_node_b else 1.0,
                    alpha=1.0 if node_a == edge_node_a and node_b == edge_node_b else 0.3)

        found += 1
    # set title for all plots
    fig.suptitle('Selected false neagtives in BitcoinAlpha and their local graph structure')
    # add manual line legends
    axes[0, 0].plot([], [], color='green', label='positive')
    axes[0, 0].plot([], [], color='red', label='negative')
    axes[0, 0].plot([], [], color='green', linestyle='dashed', label='false negative edge')
    axes[0, 0].plot([], [], color='red', linestyle='dashed', label='false positive edge')

    axes[0, 0].legend()

    plt.show()