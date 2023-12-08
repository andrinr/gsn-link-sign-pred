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

# Local dependencies
import simulation as sm
import graph as g
from io_helpers import get_dataset

def main(argv) -> None:
    """
    Main function

    Parameters:
    ----------  
    -s : int (default=64)
        Embedding dimension
    -o : int (default=0)
        Number of iterations for the optimizer
    """
    # Simulation parameters
    NEURAL_FORCE = False
    PRE_TRAIN_NEURAL_FORCE = False      
    OPTIMIZE_NEURAL_FORCE = False
    OPTIMIZE_HEURISTIC_FORCE = False
    EMBEDDING_DIM = 64
    INIT_POS_RANGE = 1.0
    TEST_DT = 0.001
    DAMPING = 0.064
    CENTERING = 0

    # Training parameters
    NUM_EPOCHS = 50
    MULTISTPES_GRADIENT = 1
    GRAPH_PARTITIONING = False
    BATCH_NUMBER = 6
    TRAIN_DT = 0.001
    PER_EPOCH_SIM_ITERATIONS = 300
    FINAL_SIM_ITERATIONS = 2048
    TEST_SHOTS = 1

    # Paths
    DATA_PATH = 'src/data/'
    CECKPOINT_PATH = 'checkpoints/'

    assert not (OPTIMIZE_NEURAL_FORCE and OPTIMIZE_HEURISTIC_FORCE), "Cannot optimize spring params and use NN force at the same time"
    assert not (not NEURAL_FORCE and OPTIMIZE_NEURAL_FORCE), "Cannot optimize force without using NN force"

    if not OPTIMIZE_NEURAL_FORCE and not OPTIMIZE_HEURISTIC_FORCE:
        NUM_EPOCHS = 0

    # Deactivate preallocation of memory to avoid OOM errors
    #os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".95"
    #os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

    #jax.config.update("jax_enable_x64", True)
    
    dataset, dataset_name = get_dataset(DATA_PATH, argv) 
    if not is_undirected(dataset.edge_index):
        transform = T.ToUndirected(reduce="min")
        dataset = transform(dataset)

    num_nodes = dataset.num_nodes
    num_edges = dataset.num_edges

    print(f"num_nodes: {num_nodes}")
    print(f"num_edges: {num_edges}")

    batches = []
    if GRAPH_PARTITIONING and (OPTIMIZE_NEURAL_FORCE or OPTIMIZE_HEURISTIC_FORCE):
        cluster_data = ClusterData(
            dataset, 
            num_parts=BATCH_NUMBER,
            keep_inter_cluster_edges=True)
        
        loader = ClusterLoader(cluster_data)
        for batch in loader:
            signedGraph = g.to_SignedGraph(batch)
            print(f"num_nodes: {signedGraph.num_nodes}")
            if signedGraph.num_nodes > 50:
                batches.append(signedGraph)

    else:
        batches.append(g.to_SignedGraph(dataset))
        
    params_path = f"{CECKPOINT_PATH}params_{EMBEDDING_DIM}.yaml"
    if os.path.exists(params_path):
        stream = open(params_path, 'r')
        heuristic_force_params = yaml.load(stream, Loader=yaml.FullLoader)
        heuristic_force_params = sm.HeuristicForceParams(**heuristic_force_params)
        print("loaded spring params checkpoint")
    else:
        heuristic_force_params = sm.HeuristicForceParams(
            friend_distance=5.0,
            friend_stiffness=0.3,
            neutral_distance=10.0,
            neutral_stiffness=0.3,
            enemy_distance=20.0,
            enemy_stiffness=0.3,
            degree_multiplier=50.0)
        print("no spring params checkpoint found, using default params")

    simulation_params_train = sm.SimulationParams(
        iterations=PER_EPOCH_SIM_ITERATIONS // MULTISTPES_GRADIENT,
        dt=TRAIN_DT,
        damping=DAMPING,
        centering=CENTERING)

    # Create initial values for neural network parameters
    key_force, key_training, key_test = random.split(random.PRNGKey(2), 3)

    # params including embdding size
    force_params_name = f"{CECKPOINT_PATH}force_params_{EMBEDDING_DIM}.yaml"
    if os.path.exists(force_params_name):
        stream = open(force_params_name, 'r')
        neural_force_params = yaml.load(stream, Loader=yaml.UnsafeLoader)
        print("loaded force params checkpoint")
    else:
        neural_force_params = sm.init_neural_force_params(
            num_dimensions=EMBEDDING_DIM,
            key=key_force,
            factor=0.1)
        print("no force params checkpoint found, using default params")
    
    if OPTIMIZE_NEURAL_FORCE and PRE_TRAIN_NEURAL_FORCE and NEURAL_FORCE:
        neural_force_params = sm.pre_train(
            key=key_training,
            learning_rate=0.1,
            num_epochs=400,
            heuristic_force_params=heuristic_force_params,
            neural_force_params=neural_force_params)
        
    force_params = neural_force_params if NEURAL_FORCE else heuristic_force_params
        
    if OPTIMIZE_HEURISTIC_FORCE or OPTIMIZE_NEURAL_FORCE:
        force_params, loss_hist, metrics_hist, force_params_hist = sm.train(
            random_key=key_training,
            batches=batches,
            force_params=force_params,
            training_params= sm.TrainingParams(
                num_epochs=NUM_EPOCHS,
                learning_rate=0.06,
                use_neural_force=NEURAL_FORCE,
                batch_size=BATCH_NUMBER,
                init_pos_range=INIT_POS_RANGE,
                embedding_dim=EMBEDDING_DIM,
                multi_step=MULTISTPES_GRADIENT),
            simulation_params=simulation_params_train)

        # plot loss over time
        epochs = range(NUM_EPOCHS)
        plt.plot(epochs, loss_hist)
        plt.title('Loss')
        plt.show()

        # plot metrics over time
        plt.plot(epochs, [metrics.auc for metrics in metrics_hist])
        plt.plot(epochs, [metrics.f1_binary for metrics in metrics_hist])
        plt.plot(epochs, [metrics.f1_micro for metrics in metrics_hist])
        plt.plot(epochs, [metrics.f1_macro for metrics in metrics_hist])
        plt.legend(['AUC', 'F1 binary', 'F1 micro', 'F1 macro'])
        plt.title('Measures')
        plt.show()

        if not NEURAL_FORCE:
            # plot piecewise linear functions
            n = 300
            x = jnp.linspace(-3, 10, n)
            x = jnp.expand_dims(x, axis=1)
            y_friend = sm.heuristic_force(
                params=force_params,
                distance=x,
                sign=jnp.ones(n))
            y_friend = y_friend.squeeze()

            y_enemy = sm.heuristic_force(
                params=force_params,
                distance=x,
                sign=jnp.ones(n) * -1)
            y_enemy = y_enemy.squeeze()

            y_neutral = sm.heuristic_force(
                params=force_params,
                distance=x,
                sign=jnp.zeros(n))
            y_neutral = y_neutral.squeeze()

            print(x.shape)
            print(y_friend.shape)

            plt.plot(x, y_friend, label='friend')
            plt.plot(x, y_enemy, label='enemy')
            plt.plot(x, y_neutral, label='neutral')
            plt.legend()
            plt.title('Piecewise linear functions')
            plt.show()
            
    # write spring params to file, the file is still a traced jax object
    if OPTIMIZE_HEURISTIC_FORCE:
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
                params_dict = {}
                # get value from jax traced array
                params_dict['friend_distance'] = force_params.friend_distance.item()
                params_dict['friend_stiffness'] = force_params.friend_stiffness.item()
                params_dict['neutral_distance'] = force_params.neutral_distance.item()
                params_dict['neutral_stiffness'] = force_params.neutral_stiffness.item()
                params_dict['enemy_distance'] = force_params.enemy_distance.item()
                params_dict['enemy_stiffness'] = force_params.enemy_stiffness.item()
                params_dict['degree_multiplier'] = force_params.degree_multiplier.item()
                
                yaml.dump(params_dict, file)

                

    # Store the trained parameters in a file
    if OPTIMIZE_NEURAL_FORCE:
        # ask user if they want to save the parameters
        questions = [
            inquirer.List('save',
                message="Do you want to save the neural network parameters?",
                choices=['Yes', 'No'],
            ),
        ]
        answers = inquirer.prompt(questions)
        if answers['save'] == 'Yes':

            with open(force_params_name, 'w') as file:
                yaml.dump(force_params, file)

    shot_metrics = []
    key_shots = random.split(key_test, TEST_SHOTS)     
    graph = {}

    for shot in range(TEST_SHOTS):

        graph = g.to_SignedGraph(dataset)
   
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

        if NEURAL_FORCE:
            spring_state = sm.simulate(
                simulation_params=simulation_params_test,
                spring_state=spring_state, 
                force_params=force_params,
                use_neural_force=NEURAL_FORCE,
                graph=training_graph)
        else:
            spring_state = sm.simulate(
                simulation_params=simulation_params_test,
                spring_state=spring_state, 
                force_params=force_params,
                use_neural_force=NEURAL_FORCE,
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

    graph = g.to_SignedGraph(dataset)
   
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

    if NEURAL_FORCE:
        spring_state = sm.simulate(
            simulation_params=simulation_params_test,
            spring_state=spring_state, 
            force_params=force_params,
            use_neural_force=NEURAL_FORCE,
            graph=training_graph)
    else:
        tree_depth = int(np.log2(EMBEDDING_DIM))
        # initialize spring state
        spring_state = sm.init_spring_state(
            rng=key_shots[shot],
            n=graph.num_nodes,
            m=graph.num_edges,
            range=INIT_POS_RANGE,
            embedding_dim=64
    	)
    
        for i in range(8):
            simulation_params_test = sm.SimulationParams(
                iterations=(i+1) * 64,
                dt=TEST_DT,
                damping=DAMPING,
                centering=CENTERING)
                
            spring_state = sm.simulate(
                simulation_params=simulation_params_test,
                spring_state=spring_state, 
                force_params=force_params,
                use_neural_force=NEURAL_FORCE,
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

    plt.scatter(graph.degree.values, error_per_node)
    plt.colorbar()
    plt.xlabel('node degree')
    plt.ylabel('predicted sign')
    plt.title('predicted sign per node degree')
    plt.show()
    

    
if __name__ == "__main__":
    main(sys.argv[1:])
    