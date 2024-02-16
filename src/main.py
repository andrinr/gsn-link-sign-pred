# External dependencies
import sys
import inquirer
from torch_geometric.utils import is_undirected
from torch_geometric.loader import ClusterData, ClusterLoader, GraphSAINTSampler
import yaml
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import torch_geometric.transforms as T
import numpy as np
import jax
import torch

# Local dependencies
import simulation as sm
import graph as g
from io_helpers import get_dataset
import stats as stats

def main(argv) -> None:
    """
    Main function
    """
    # Simulation parameters
    OPTIMIZE_FORCE_PARAMS = True
    TREAT_AS_UNDIRECTED = True
    EMBEDDING_DIM = 20
    INIT_POS_RANGE = 1.0
    TEST_DT = 0.001
    DAMPING = 0.03
    CENTERING = 0.15

    # Training parameters
    NUM_EPOCHS = 100
    MULTISTPES_GRADIENT = 1
    GRAPH_PARTITIONING = False
    BATCH_NUMBER = 10
    TRAIN_DT = 0.002
    PER_EPOCH_SIM_ITERATIONS = 200
    FINAL_SIM_ITERATIONS = 600
    TEST_SHOTS = 5

    N_EPOCH_SEQUENCE = [50, 40, 30, 20, 20, 20, 40]
    TRAIN_ITER_SEQUENCE = [50, 60, 70, 80, 100, 150, 200]
    DT_SEQUENCE = [0.01, 0.01, 0.005, 0.005, 0.005, 0.002, 0.002]
    DAMPING_SEQUENCE = [0.2, 0.1, 0.05, 0.05, 0.03, 0.03, 0.03]

    # Paths
    DATA_PATH = 'src/data/'
    CECKPOINT_PATH = 'checkpoints/'

    # Deactivate preallocation of memory to avoid OOM errors
    #os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".95"
    #os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

    jax.config.update("jax_enable_x64", True)
    
    dataset, dataset_name = get_dataset(DATA_PATH, argv) 
    if not is_undirected(dataset.edge_index) and TREAT_AS_UNDIRECTED:
        transform = T.ToUndirected(reduce="min")
        dataset = transform(dataset)

    num_nodes = dataset.num_nodes
    num_edges = dataset.num_edges

    print(f"num_nodes: {num_nodes}")
    print(f"num_edges: {num_edges}")

    s = stats.Triplets(dataset)(n_triplets=3000, seed=0)
    print(s.p_balanced)

    print(dataset.edge_attr)
    print(f"percentage of positive edges: {torch.sum(dataset.edge_attr == 1) / dataset.num_edges}")


    batches = []
    if GRAPH_PARTITIONING and OPTIMIZE_FORCE_PARAMS:
        cluster_data = ClusterData(
            dataset, 
            num_parts=BATCH_NUMBER,
            keep_inter_cluster_edges=True)
        
        loader = ClusterLoader(cluster_data)
        for batch in loader:
            signedGraph = g.to_SignedGraph(batch, TREAT_AS_UNDIRECTED)
            print(f"num_nodes: {signedGraph.num_nodes}")
            if signedGraph.num_nodes > 50:
                batches.append(signedGraph)

    else:
        batches.append(g.to_SignedGraph(dataset, TREAT_AS_UNDIRECTED))


    params_path = f"{CECKPOINT_PATH}params_{EMBEDDING_DIM}.yaml"
    if os.path.exists(params_path):
        stream = open(params_path, 'r')
        heuristic_force_params = yaml.load(stream,  Loader=yaml.UnsafeLoader)
        # heuristic_force_params = sm.HeuristicForceParams(**heuristic_force_params)
        print("loaded spring params checkpoint")
    else:
        heuristic_force_params = sm.HeuristicForceParams(
            friend=sm.MLP(
                w0=jax.random.normal(random.PRNGKey(0), (7, 4)),
                b0=jnp.zeros(4),
                w1=jax.random.normal(random.PRNGKey(1), (4,1)),
                b1=jnp.zeros(1)),
            neutral=sm.MLP(
                w0=jax.random.normal(random.PRNGKey(2), (7, 4)),
                b0=jnp.zeros(4),
                w1=jax.random.normal(random.PRNGKey(3), (4,1)),
                b1=jnp.zeros(1)),
            enemy=sm.MLP(
                w0=jax.random.normal(random.PRNGKey(4), (7, 4)),
                b0=jnp.zeros(4),
                w1=jax.random.normal(random.PRNGKey(5), (4,1)),
                b1=jnp.zeros(1)),
        )

        print("no spring params checkpoint found, using default params")

    # Create initial values for neural network parameters
    key_force, key_training, key_test = random.split(random.PRNGKey(2), 3)

    force_params = heuristic_force_params

    if OPTIMIZE_FORCE_PARAMS:
        loss_hist = []
        metrics_hist = []
        force_params_hist = []
        dt_hist = []
        damping_hist = []
        iterations_hist = []


        for settings in zip(N_EPOCH_SEQUENCE, TRAIN_ITER_SEQUENCE, DT_SEQUENCE, DAMPING_SEQUENCE):
            NUM_EPOCHS, PER_EPOCH_SIM_ITERATIONS, TRAIN_DT, DAMPING = settings

            simulation_params_train = sm.SimulationParams(
                iterations=PER_EPOCH_SIM_ITERATIONS,
                dt=TRAIN_DT,
                damping=DAMPING,
                centering=CENTERING)

            force_params, loss_hist_, metrics_hist_, force_params_hist_ = sm.train(
                random_key=key_training,
                batches=batches,
                force_params=force_params,
                training_params= sm.TrainingParams(
                    num_epochs=NUM_EPOCHS,
                    learning_rate=0.04,
                    batch_size=BATCH_NUMBER,
                    init_pos_range=INIT_POS_RANGE,
                    embedding_dim=EMBEDDING_DIM,
                    multi_step=MULTISTPES_GRADIENT),
                simulation_params=simulation_params_train)
            
            #concatenate lists
            loss_hist = loss_hist + loss_hist_
            metrics_hist = metrics_hist + metrics_hist_
            force_params_hist = force_params_hist + force_params_hist_

            dt_hist = dt_hist + list(jnp.ones(NUM_EPOCHS) * TRAIN_DT)
            iterations_hist = iterations_hist + list(jnp.ones(NUM_EPOCHS) * PER_EPOCH_SIM_ITERATIONS)
            damping_hist = damping_hist + list(jnp.ones(NUM_EPOCHS) * DAMPING)

        # plot loss over time
        epochs = np.arange(0, len(loss_hist))
        plt.plot(epochs, loss_hist)
        plt.title('Loss')
        plt.show()

        # get set1 color map
        colors = plt.cm.get_cmap('Set1', 10)

        # Three subplots with shared x axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        # plot metrics over time
        ax1.plot(epochs, [metrics.auc for metrics in metrics_hist], color=colors(0))
        ax1.plot(epochs, [metrics.f1_macro for metrics in metrics_hist], color=colors(1))
        ax1.legend(['AUC', 'F1 macro'], loc='lower left')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')

        ax2.plot(epochs, dt_hist, color=colors(2))
        ax2.plot(epochs, damping_hist, color=colors(3))

        ax2.legend(['dt', 'damping'], loc='lower left')
        ax2.set_xlabel('Epochs')

        ax3.plot(epochs, iterations_hist, color=colors(4))
        ax3.legend(['iterations'], loc='lower left')
        ax3.set_xlabel('Epochs')

        plt.show()

    # write spring params to file, the file is still a traced jax object
    if OPTIMIZE_FORCE_PARAMS:
        # ask user if they want to save the parameters
        questions = [
            inquirer.List('save',
                message="Do you want to save the spring parameters?",
                choices=['Yes', 'No'],
            ),
        ]
        answers = inquirer.prompt(questions)
        if answers['save'] == 'Yes':
            with open(params_path, 'w') as file:
                yaml.dump(force_params, file)   

    shot_metrics = []
    key_shots = random.split(key_test, TEST_SHOTS)     
    graph = {}

    for shot in range(TEST_SHOTS):

        graph = g.to_SignedGraph(dataset, TREAT_AS_UNDIRECTED)
   
        # initialize spring state
        spring_state = sm.init_spring_state(
            rng=key_shots[shot],
            n=graph.num_nodes,
            m=graph.num_edges,
            range=INIT_POS_RANGE,
            embedding_dim=EMBEDDING_DIM
        )

        # training_signs = graphsigns.copy()
        # training_signs = training_signs.at[train_mask].set(0)


        simulation_params_test = sm.SimulationParams(
            iterations=FINAL_SIM_ITERATIONS,
            dt=TEST_DT,
            damping=DAMPING,
            centering=CENTERING)

        training_signs = graph.sign.copy()
        training_signs = jnp.where(graph.train_mask, training_signs, 0)
        training_graph = graph._replace(sign=training_signs)
        training_signs_one_hot = jax.nn.one_hot(training_signs + 1, 3)
        training_graph = training_graph._replace(sign_one_hot=training_signs_one_hot)

       
        spring_state = sm.simulate(
            simulation_params=simulation_params_test,
            spring_state=spring_state, 
            force_params=force_params,
            graph=training_graph)

        metrics, _ = sm.evaluate(
            spring_state,
            graph.edge_index,
            graph.sign,
            graph.train_mask,
            graph.test_mask)
        
        shot_metrics.append(metrics)

    # print average metrics over all shots
    print(f"average metrics over {TEST_SHOTS} shots:")
    print(f"auc: {np.mean([metrics.auc for metrics in shot_metrics])}")
    print(f"f1_binary: {np.mean([metrics.f1_binary for metrics in shot_metrics])}")
    print(f"f1_micro: {np.mean([metrics.f1_micro for metrics in shot_metrics])}")
    print(f"f1_macro: {np.mean([metrics.f1_macro for metrics in shot_metrics])}")
    print(f"true_positives: {np.mean([metrics.true_positives for metrics in shot_metrics])}")
    print(f"false_positives: {np.mean([metrics.false_positives for metrics in shot_metrics])}")
    print(f"true_negatives: {np.mean([metrics.true_negatives for metrics in shot_metrics])}")
    print(f"false_negatives: {np.mean([metrics.false_negatives for metrics in shot_metrics])}")

    # standard deviation
    print(f"std auc: {np.std([metrics.auc for metrics in shot_metrics])}")
    print(f"std f1_binary: {np.std([metrics.f1_binary for metrics in shot_metrics])}")
    print(f"std f1_micro: {np.std([metrics.f1_micro for metrics in shot_metrics])}")
    print(f"std f1_macro: {np.std([metrics.f1_macro for metrics in shot_metrics])}")
    print(f"std true_positives: {np.std([metrics.true_positives for metrics in shot_metrics])}")
    print(f"std false_positives: {np.std([metrics.false_positives for metrics in shot_metrics])}")
    print(f"std true_negatives: {np.std([metrics.true_negatives for metrics in shot_metrics])}")
    print(f"std false_negatives: {np.std([metrics.false_negatives for metrics in shot_metrics])}")  

    # print extreme metrics over all shots
    print(f"extreme metrics over {TEST_SHOTS} shots:")
    print(f"auc: {np.max([metrics.auc for metrics in shot_metrics])}")
    print(f"f1_binary: {np.max([metrics.f1_binary for metrics in shot_metrics])}")
    print(f"f1_micro: {np.max([metrics.f1_micro for metrics in shot_metrics])}")
    print(f"f1_macro: {np.max([metrics.f1_macro for metrics in shot_metrics])}")
    print(f"true_positives: {np.max([metrics.true_positives for metrics in shot_metrics])}")
    print(f"false_positives: {np.max([metrics.false_positives for metrics in shot_metrics])}")
    print(f"true_negatives: {np.max([metrics.true_negatives for metrics in shot_metrics])}")
    print(f"false_negatives: {np.max([metrics.false_negatives for metrics in shot_metrics])}")

    # selected_wrong_classification(
    #     dataset=dataset,
    #     key=key_test,
    #     spring_params=spring_params,
    #     init_range=INIT_POS_RANGE,
    #     dim=EMBEDDING_DIM,
    #     iterations=FINAL_SIM_ITERATIONS,
    #     dt=TEST_DT,
    #     damping=DAMPING,
    # )

    graph = g.to_SignedGraph(dataset, TREAT_AS_UNDIRECTED)
   
    # initialize spring state
    spring_state = sm.init_spring_state(
        rng=key_shots[shot],
        n=graph.num_nodes,
        m=graph.num_edges,
        range=INIT_POS_RANGE,
        embedding_dim=EMBEDDING_DIM
    )

    # training_signs = graphsigns.copy()
    # training_signs = training_signs.at[train_mask].set(0)


    simulation_params_test = sm.SimulationParams(
        iterations=FINAL_SIM_ITERATIONS,
        dt=TEST_DT,
        damping=DAMPING,
        centering=CENTERING)

    training_signs = graph.sign.copy()
    training_signs = jnp.where(graph.train_mask, training_signs, 0)
    training_graph = graph._replace(sign=training_signs)
    training_signs_one_hot = jax.nn.one_hot(training_signs + 1, 3)
    training_graph = training_graph._replace(sign_one_hot=training_signs_one_hot)


    # initialize spring state
    spring_state = sm.init_spring_state(
        rng=key_shots[shot],
        n=graph.num_nodes,
        m=graph.num_edges,
        range=INIT_POS_RANGE,
        embedding_dim=64
    )

    for i in range(16):
        simulation_params_test = sm.SimulationParams(
            iterations=(i+1) * 64,
            dt=TEST_DT,
            damping=DAMPING,
            centering=CENTERING)
            
        spring_state = sm.simulate(
            simulation_params=simulation_params_test,
            spring_state=spring_state, 
            force_params=force_params,
            graph=training_graph)

        metrics = sm.evaluate(
            spring_state,
            graph.edge_index,
            graph.sign,
            graph.train_mask,
            graph.test_mask)
        
        print(metrics)

    metrics = sm.evaluate(
        spring_state,
        graph.edge_index,
        graph.sign,
        graph.train_mask,
        graph.test_mask)
    
    print(metrics)

    # plot error per edge against avg node distance to center
    embedding_i = spring_state.position[graph.edge_index[0]]
    embedding_j = spring_state.position[graph.edge_index[1]]
    distance_i = jnp.linalg.norm(embedding_i, axis=1, keepdims=True)
    distance_j = jnp.linalg.norm(embedding_j, axis=1, keepdims=True)

    avg_distance = (distance_i + distance_j) / 2
    distance = jnp.linalg.norm(embedding_j - embedding_i, axis=1, keepdims=True)

    predicted_sign = sm.predict(
        spring_state = spring_state,
        graph = graph,
        x_0 = 0)
    print(graph.sign)
    print(predicted_sign)

    sign = jnp.where(graph.sign == 1, 1, 0)
    print(sign)
    sign_pred = jnp.where(predicted_sign > 0.01, 1, 0)
    print(sign_pred)

    true_positives = jnp.logical_and(sign == 1, sign_pred == 1)
    false_positives = jnp.logical_and(sign == 0, sign_pred == 1)
    true_negatives = jnp.logical_and(sign == 0, sign_pred == 0)
    false_negatives = jnp.logical_and(sign == 1, sign_pred == 0)

    print(f"true_positives: {jnp.sum(true_positives)}")
    print(f"false_positives: {jnp.sum(false_positives)}")
    print(f"true_negatives: {jnp.sum(true_negatives)}")
    print(f"false_negatives: {jnp.sum(false_negatives)}")

    print(f"sign: {sign}")
    print(f"sign_pred: {sign_pred}")

    error = jnp.square(sign - predicted_sign)

    plt.scatter(distance, error, c=predicted_sign, cmap='viridis', s=0.5)
    plt.colorbar()
    plt.xlabel('avg distance to center')
    plt.ylabel('error')
    plt.title('error per edge')
    plt.show()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(spring_state.position)
    embedding = pca.transform(spring_state.position)

    plt.scatter(embedding[:,0], embedding[:,1], cmap='viridis', s=0.5)
    plt.colorbar()
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('PCA embedding')
    plt.show()

    error_per_node = jnp.zeros(graph.num_nodes)
    error_per_node = error_per_node.at[graph.edge_index[0]].add(error)
    error_per_node = jnp.expand_dims(error_per_node, axis=1)
    # normalize error by node degree, elementwise division
    error_per_node = error_per_node / graph.degree.values

    print(f"error_per_node: {error_per_node.shape}")

    plt.scatter(graph.degree.values, error_per_node, s=1.5)
    plt.xlabel('node degree')
    plt.ylabel('error')
    # plt.title(')
    plt.show()
    

    
if __name__ == "__main__":
    main(sys.argv[1:])
    