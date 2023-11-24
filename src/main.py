# External dependencies
import sys
import inquirer
from torch_geometric.utils import is_undirected
from torch_geometric.loader import ClusterData, ClusterLoader
import yaml
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import torch_geometric.transforms as T
import numpy as np

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
    NEURAL_FORCE = True
    PRE_TRAIN_NEURAL_FORCE = True      
    OPTIMIZE_NEURAL_FORCE = True
    OPTIMIZE_HEURISTIC_FORCE = False
    EMBEDDING_DIM = 16
    INIT_POS_RANGE = 3.0
    TEST_DT = 0.0025
    DAMPING = 0.032

    # Training parameters
    NUM_EPOCHS = 30
    MULTISTPES_GRADIENT = 3
    GRAPH_PARTITIONING = True
    BATCH_SIZE = 12
    TRAIN_DT = 0.005
    PER_EPOCH_SIM_ITERATIONS = 300
    FINAL_SIM_ITERATIONS = 2048
    AUXILLARY_ITERATIONS = 4
    TEST_SHOTS = 10

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
    
    dataset = get_dataset(DATA_PATH, argv) 
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
            num_parts=BATCH_SIZE)
        
        loader = ClusterLoader(cluster_data)
        for batch in loader:
            signedGraph = g.to_SignedGraph(batch)
            if signedGraph.num_nodes > 300:
                batches.append(signedGraph)
            
    else:
        signedGraph = g.to_SignedGraph(dataset)
        batches.append(signedGraph)
        
    params_path = f"{CECKPOINT_PATH}params_{EMBEDDING_DIM}.yaml"
    if os.path.exists(params_path):
        stream = open(params_path, 'r')
        spring_params = yaml.load(stream, Loader=yaml.FullLoader)
        spring_params = sm.HeuristicForceParams(**spring_params)
        print("loaded spring params checkpoint")
    else:
        spring_params = sm.HeuristicForceParams(
            friend_distance=5.0,
            friend_stiffness=0.3,
            neutral_distance=10.0,
            neutral_stiffness=0.3,
            enemy_distance=20.0,
            enemy_stiffness=0.3,
            distance_threshold=4.0,
            center_attraction=0.0)
        print("no spring params checkpoint found, using default params")

    simulation_params_train = sm.SimulationParams(
        iterations=PER_EPOCH_SIM_ITERATIONS // MULTISTPES_GRADIENT,
        dt=TRAIN_DT,
        damping=DAMPING,
        message_passing_iterations=AUXILLARY_ITERATIONS)

    # Create initial values for neural network parameters
    key_force, key_training, key_test = random.split(random.PRNGKey(2), 3)

    # params including embdding size
    force_params_name = f"{CECKPOINT_PATH}force_params_{EMBEDDING_DIM}.yaml"
    if os.path.exists(force_params_name):
        stream = open(force_params_name, 'r')
        force_params = yaml.load(stream, Loader=yaml.UnsafeLoader)
        print("loaded force params checkpoint")
    else:
        force_params = sm.init_neural_force_params(
            key=key_force,
            factor=0.1)
        print("no force params checkpoint found, using default params")
    
    if OPTIMIZE_NEURAL_FORCE and PRE_TRAIN_NEURAL_FORCE:
        force_params = sm.pre_train(
            key=key_training,
            learning_rate=1e-2,
            num_epochs=200,
            heuristic_force_params=spring_params,
            neural_force_params=force_params)
        
    if OPTIMIZE_HEURISTIC_FORCE or OPTIMIZE_NEURAL_FORCE:
        force_params, loss_hist, metrics_hist = sm.train(
            random_key=key_training,
            batches=batches,
            force_params=force_params,
            training_params= sm.TrainingParams(
                num_epochs=NUM_EPOCHS,
                learning_rate=1e-3,
                use_neural_force=NEURAL_FORCE,
                batch_size=BATCH_SIZE,
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
                params_dict['friend_distance'] = spring_params.friend_distance.item()
                params_dict['friend_stiffness'] = spring_params.friend_stiffness.item()
                params_dict['neutral_distance'] = spring_params.neutral_distance.item()
                params_dict['neutral_stiffness'] = spring_params.neutral_stiffness.item()
                params_dict['enemy_distance'] = spring_params.enemy_distance.item()
                params_dict['enemy_stiffness'] = spring_params.enemy_stiffness.item()
                params_dict['distance_threshold'] = spring_params.distance_threshold.item()
                params_dict['center_attraction'] = spring_params.center_attraction.item()

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
    for shot in range(TEST_SHOTS):

        graph = g.to_SignedGraph(dataset)
        print(graph.sign)

        print(f"shot: {shot}")

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
            message_passing_iterations=AUXILLARY_ITERATIONS)

        training_signs = graph.sign.copy()
        training_signs = jnp.where(graph.train_mask, training_signs, 0)
        training_graph = graph._replace(sign=training_signs)

        spring_state = sm.simulate(
            simulation_params=simulation_params_test,
            spring_state=spring_state, 
            force_params=spring_params,
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

    # print extreme metrics over all shots
    print(f"extreme metrics over {TEST_SHOTS} shots:")
    print(f"auc: {np.max([metrics.auc for metrics in shot_metrics])}")
    print(f"f1_binary: {np.max([metrics.f1_binary for metrics in shot_metrics])}")
    print(f"f1_micro: {np.max([metrics.f1_micro for metrics in shot_metrics])}")
    print(f"f1_macro: {np.max([metrics.f1_macro for metrics in shot_metrics])}")

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

    
if __name__ == "__main__":
    main(sys.argv[1:])
    