# External dependencies
import sys
import inquirer
from torch_geometric.utils import is_undirected, subgraph
from torch_geometric.loader import ClusterData, ClusterLoader
import yaml
from jax import random, value_and_grad
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import optax
import os
import torch_geometric.transforms as T
import numpy as np
from sklearn.decomposition import PCA

# Local dependencies
import simulation as sim
import neural as nn
from io_helpers import get_dataset
from plot_helpers import plot_embedding, selected_wrong_classification
from graph import to_SignedGraph

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
    NN_FORCE = False
    OPTIMIZE_FORCE = False
    OPTIMIZE_SPRING_PARAMS = False
    EMBEDDING_DIM = 64
    INIT_POS_RANGE = 3.0
    TEST_DT = 0.0025
    DAMPING = 0.032

    # Training parameters
    NUM_EPOCHS = 15
    GRADIENT_ACCUMULATION = 1
    SUBSTEPS = 1
    BATCH_SIZE = 4
    TRAIN_DT = 0.005
    PER_EPOCH_SIM_ITERATIONS = 300
    FINAL_SIM_ITERATIONS = 2048
    AUXILLARY_ITERATIONS = 4
    GRAPH_PARTITIONING = True

    TEST_SHOTS = 10

    # Paths
    DATA_PATH = 'src/data/'
    CECKPOINT_PATH = 'checkpoints/'

    assert not (OPTIMIZE_FORCE and OPTIMIZE_SPRING_PARAMS), "Cannot optimize spring params and use NN force at the same time"
    assert not (not NN_FORCE and OPTIMIZE_FORCE), "Cannot optimize force without using NN force"

    if not OPTIMIZE_FORCE and not OPTIMIZE_SPRING_PARAMS:
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
    if GRAPH_PARTITIONING and (OPTIMIZE_FORCE or OPTIMIZE_SPRING_PARAMS):
        cluster_data = ClusterData(
            dataset, 
            num_parts=BATCH_SIZE)
        
        loader = ClusterLoader(cluster_data)
        for batch in loader:
            signedGraph = to_SignedGraph(batch)
            if signedGraph.num_nodes > 300:
                batches.append(signedGraph)
            
    else:
        signedGraph = to_SignedGraph(dataset)
        batches.append(signedGraph)

        
    params_path = f"{CECKPOINT_PATH}params_{EMBEDDING_DIM}.yaml"
    if os.path.exists(params_path):
        stream = open(params_path, 'r')
        spring_params = yaml.load(stream, Loader=yaml.FullLoader)
        spring_params = sim.SpringParams(**spring_params)
        print("loaded spring params checkpoint")
    else:
        spring_params = sim.SpringParams(
            friend_distance=5.0,
            friend_stiffness=0.3,
            neutral_distance=10.0,
            neutral_stiffness=0.3,
            enemy_distance=20.0,
            enemy_stiffness=0.3,
            distance_threshold=4.0,
            center_attraction=0.0)
        print("no spring params checkpoint found, using default params")

    simulation_params_train = sim.SimulationParams(
        iterations=PER_EPOCH_SIM_ITERATIONS // GRADIENT_ACCUMULATION,
        dt=TRAIN_DT,
        damping=DAMPING,
        message_passing_iterations=AUXILLARY_ITERATIONS)

    # Create initial values for neural network parameters
    key_force, key_training, key_test, key_offset = random.split(random.PRNGKey(2), 4)

    # params including embdding size
    force_params_name = f"{CECKPOINT_PATH}force_params_{EMBEDDING_DIM}.yaml"
    if os.path.exists(force_params_name):
        stream = open(force_params_name, 'r')
        force_params = yaml.load(stream, Loader=yaml.UnsafeLoader)
        print("loaded force params checkpoint")
    else:
        force_params = nn.init_force_params(
            key=key_force,
            factor=0.1)
        print("no force params checkpoint found, using default params")
    
    # setup optax optimizers
    if OPTIMIZE_FORCE:
        force_optimizer = optax.adam(learning_rate=1e-1)
        force_multi_step = optax.MultiSteps(force_optimizer, GRADIENT_ACCUMULATION)
        force_optimizier_state = force_multi_step.init(force_params)

        value_grad_fn = value_and_grad(sim.simulate_and_loss, argnums=4, has_aux=True)

    if OPTIMIZE_SPRING_PARAMS:
        params_optimizer = optax.adamaxw(learning_rate=1e-3)
        params_multi_step = optax.MultiSteps(params_optimizer, GRADIENT_ACCUMULATION)
        params_optimizier_state = params_multi_step.init(spring_params)  

        value_grad_fn = value_and_grad(sim.simulate_and_loss, argnums=2, has_aux=True)

    loss_hist = []
    metrics_hist = []
    spring_hist = []    

    epoch_loss = 0
    epoch_correct = 0
    epoch_num_total = 0

    epochs = range(NUM_EPOCHS)

    epoch_score = 0

    epochs_keys = random.split(key_training, NUM_EPOCHS)
    for epoch_index in epochs:
        for batch_index, batch_graph in enumerate(batches):

            print(f"EPOCH: #{epoch_index} BATCH: #{batch_index}")

            # initialize spring state
            # take new key each time to avoid overfitting to specific initial conditions
            spring_state = sim.init_spring_state(
                rng=epochs_keys[epoch_index],
                range=INIT_POS_RANGE,
                n=batch_graph.num_nodes,
                m=batch_graph.num_edges,
                embedding_dim=EMBEDDING_DIM)

            # run simulation and compute loss, auxillaries and gradient
            for i in range(SUBSTEPS):
                (loss_value, (spring_state, signs_pred)), grad = value_grad_fn(
                    simulation_params_train, #0
                    spring_state, #1
                    spring_params, #2
                    NN_FORCE, #3
                    force_params, #4
                    batch_graph)

                if OPTIMIZE_FORCE:
                    nn_force_update, force_optimizier_state = force_multi_step.update(
                        grad, force_optimizier_state, force_params)
                    
                    force_params = optax.apply_updates(force_params, nn_force_update)

                if OPTIMIZE_SPRING_PARAMS:
                    params_grad = grad
                    params_update, params_optimizier_state = params_multi_step.update(
                        params_grad, params_optimizier_state, spring_params)
                    
                    spring_params = optax.apply_updates(spring_params, params_update)

                print(f"loss: {loss_value}")

            epoch_loss += loss_value

            sign_ = batch_graph.sign * 0.5 + 0.5
            epoch_correct += jnp.sum(jnp.equal(jnp.round(signs_pred), sign_))
            epoch_num_total += batch_graph.sign.shape[0]

            print(f"loss: {loss_value}")
            metrics = sim.evaluate(
                spring_state,

                batch_graph.edge_index,
                batch_graph.sign,
                batch_graph.train_mask,
                batch_graph.test_mask)
            
            score = metrics.auc + metrics.f1_binary + metrics.f1_micro + metrics.f1_macro - loss_value
            epoch_score += score

            print(f"metrics: {metrics}")

        print(f"epoch: {epoch_index}")
        # print(f"predictions: {signs_pred}")
        print(f"epoch score: {epoch_score}")
        print(f"epoch loss: {epoch_loss}")
        print(f"epoch correct percentage: {epoch_correct / epoch_num_total}")

        loss_hist.append(epoch_loss)
        epoch_loss = 0
        epoch_score = 0
        metrics_hist.append(metrics)

        if OPTIMIZE_SPRING_PARAMS:
            spring_hist.append(spring_params)

    if OPTIMIZE_SPRING_PARAMS or OPTIMIZE_FORCE:
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

    # plot spring params over time
    if OPTIMIZE_SPRING_PARAMS:
        plt.plot(epochs, [spring_params.friend_distance for spring_params in spring_hist])
        plt.plot(epochs, [spring_params.friend_stiffness for spring_params in spring_hist])
        plt.plot(epochs, [spring_params.neutral_distance for spring_params in spring_hist])
        plt.plot(epochs, [spring_params.neutral_stiffness for spring_params in spring_hist])
        plt.plot(epochs, [spring_params.enemy_distance for spring_params in spring_hist])
        plt.plot(epochs, [spring_params.enemy_stiffness for spring_params in spring_hist])
        plt.plot(epochs, [spring_params.distance_threshold for spring_params in spring_hist])

        plt.legend(['friend_distance', 'friend_stiffness', 'neutral_distance', 'neutral_stiffness', 'enemy_distance', 'enemy_stiffness', 'distance_threshold'])
        plt.title('Spring params')
        plt.show()

    # write spring params to file, the file is still a traced jax object
    if OPTIMIZE_SPRING_PARAMS:
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
    if OPTIMIZE_FORCE:
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

        graph = to_SignedGraph(dataset)
        print(graph.sign)

        print(f"shot: {shot}")

        # initialize spring state
        spring_state = sim.init_spring_state(
            rng=key_shots[shot],
            n=graph.num_nodes,
            m=graph.num_edges,
            range=INIT_POS_RANGE,
            embedding_dim=EMBEDDING_DIM
        )

        # training_signs = graphsigns.copy()
        # training_signs = training_signs.at[train_mask].set(0)

        simulation_params_test = sim.SimulationParams(
            iterations=FINAL_SIM_ITERATIONS,
            dt=TEST_DT,
            damping=DAMPING,
            message_passing_iterations=AUXILLARY_ITERATIONS)

        training_signs = graph.sign.copy()
        training_signs = jnp.where(graph.train_mask, training_signs, 0)
        training_graph = graph._replace(sign=training_signs)

        spring_state = sim.simulate(
            simulation_params=simulation_params_test,
            spring_state=spring_state, 
            spring_params=spring_params,
            nn_force=NN_FORCE,
            nn_force_params=force_params,
            graph=training_graph)

        metrics, _ = sim.evaluate(
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
    