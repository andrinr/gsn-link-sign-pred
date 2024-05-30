# External dependencies
import sys
import inquirer
from torch_geometric.utils import is_undirected
from torch_geometric.loader import ClusterData, ClusterLoader
import yaml
from jax import random
from jax.lib import xla_bridge
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import torch_geometric.transforms as T
import numpy as np
import jax
from timeit import default_timer as timer
import pandas as pd

# Local dependencies
import simulation as sm
import graph as g
from io_helpers import get_dataset
import stats as stats
from plot_helpers import plot_embedding

def main(argv) -> None:
    """
    Main function
    """

    EMBEDDING_DIM = 20

    # Paths
    DATA_PATH = 'src/data/'
    CECKPOINT_PATH = 'training_checkpoints/'
    SCHEDULE_PATH = 'schedule/'
    OUTPUT_PATH = 'output/'

    # Define the range of the random distribution for the initial positions
    INIT_POS_RANGE = 1.0
    TEST_SHOTS = 5

    # TRAINING PARAMETERS
    MULTISTPES_GRADIENT = 1
    GRAPH_PARTITIONING = False

    # Deactivate preallocation of memory to avoid OOM errors
    #os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".95"
    #os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

    # Set the precision to 64 bit in case of NaN gradients
    jax.config.update("jax_enable_x64", True)
    print(xla_bridge.get_backend().platform)
    # disable jit compilation for debugging
    jax.config.update("jax_disable_jit", False)

    questions = [inquirer.Checkbox('multiples',
        message="Select the options: (Press <space> to select, Enter when finished).",
        choices=['Use Neural Force (SE-NN)', 'Train Parameters', 'Convert to undirected', 'Partition Graph'],
        default=['Convert to undirected']),
    ]

    answers = inquirer.prompt(questions)
    
    # read answers from inquirer
    train_parameters = False
    use_neural_froce = False
    convert_to_undirected = False
    graph_partitioning = False
    number_of_subgraphs = 10
    if 'Train Parameters' in answers['multiples']:
        train_parameters = True
    if 'Use Neural Force (SE-NN)' in answers['multiples']:
        use_neural_froce = True
    if 'Convert to undirected' in answers['multiples']:
        convert_to_undirected = True
    if 'Partition Graph' in answers['multiples']:
        graph_partitioning = True

    if graph_partitioning:
        questions = [inquirer.Text('n_subgraphs',
            message="Enter the number of subgraphs",
            default='10'),
        ]
        answers = inquirer.prompt(questions)
        number_of_subgraphs = int(answers['n_subgraphs'])
    
    dataset, dataset_name = get_dataset(DATA_PATH, argv) 
    if not is_undirected(dataset.edge_index) and convert_to_undirected:
        transform = T.ToUndirected(reduce="min")
        dataset = transform(dataset)

    # uncomment to print dataset information
    # num_nodes = dataset.num_nodes
    # num_edges = dataset.num_edges
    # s = stats.Triplets(dataset)(n_triplets=3000, seed=0)
    # print(f"num_nodes: {num_nodes}")
    # print(f"num_edges: {num_edges}")
    # print(s.p_balanced)
    # print(dataset.edge_attr)
    # print(f"percentage of positive edges: {torch.sum(dataset.edge_attr == 1) / dataset.num_edges}")
        
    force_params_path = f"{CECKPOINT_PATH}{'neural' if use_neural_froce else 'spring'}_params_{EMBEDDING_DIM}.yaml"
    print(force_params_path)
    load_checkpoint = False
    if os.path.exists(force_params_path):
        questions = [
            inquirer.List('load',
                message=f"We found a checkpoint for {'neural' if use_neural_froce else 'spring'} force parameters. Do you want to load it?",
                choices=['Yes', 'No'],
            )]
        
        answers = inquirer.prompt(questions)
        if answers['load'] == 'Yes':
            load_checkpoint = True

    # Create initial values for neural network parameters
    key_params, key_training, key_test = random.split(random.PRNGKey(2), 3)

    batches = []
    if GRAPH_PARTITIONING and train_parameters:
        cluster_data = ClusterData(
            dataset, 
            num_parts=number_of_subgraphs,
            keep_inter_cluster_edges=True)
        
        loader = ClusterLoader(cluster_data)
        for batch in loader:
            signedGraph = g.to_SignedGraph(batch, convert_to_undirected)
            print(f"num_nodes: {signedGraph.num_nodes}")
            if signedGraph.num_nodes > 50:
                batches.append(signedGraph)
    else:
        batches.append(g.to_SignedGraph(dataset, convert_to_undirected))
        
    if os.path.exists(force_params_path) and load_checkpoint:
        stream = open(force_params_path, 'r')
        force_params = yaml.load(stream,  Loader=yaml.UnsafeLoader)
    elif use_neural_froce:
        force_params = sm.init_neural_params(key_params)
    else:
        force_params = sm.init_spring_force_params()
            
    schedule_params_path = f"{SCHEDULE_PATH}{'neural' if use_neural_froce else 'spring'}_schedule.yaml"
    if os.path.exists(schedule_params_path):
        stream = open(schedule_params_path, 'r')
        schedule_params = yaml.load(stream,  Loader=yaml.UnsafeLoader)
    elif train_parameters:
        raise ValueError("no training schedule params found")


    if train_parameters:
        loss_hist = []
        metrics_hist = []
        force_params_hist = []
        dt_hist = []
        damping_hist = []
        iterations_hist = []

        for index, settings in enumerate(zip(
            schedule_params['number_of_simulations'],
            schedule_params['simulation_iterations'],
            schedule_params['simulation_dt'],
            schedule_params['simulation_damping'])):

            number_of_simulations = settings[0]
            iterations = settings[1]
            dt = settings[2]
            damping = settings[3]

            simulation_params_train = sm.SimulationParams(iterations=iterations, dt=dt, damping=damping)

            print(f"Training run {index + 1} of {len(schedule_params['number_of_simulations'])}")
            print(f"Running {number_of_simulations} simulations with {iterations} iterations each")
            print(f"Simulation parameters are dt: {dt}, damping: {damping}")

            force_params, loss_hist_, metrics_hist_, force_params_hist_ = sm.train(
                random_key=key_training,
                batches=batches,
                use_neural_force=use_neural_froce,
                force_params=force_params,
                training_params= sm.TrainingParams(
                    num_epochs=number_of_simulations,
                    learning_rate=schedule_params['learning_rate'],
                    batch_size=number_of_subgraphs,
                    init_pos_range=INIT_POS_RANGE,
                    embedding_dim=EMBEDDING_DIM,
                    multi_step=MULTISTPES_GRADIENT),
                simulation_params=simulation_params_train)
            
            loss_hist = loss_hist + loss_hist_
            metrics_hist = metrics_hist + metrics_hist_
            force_params_hist = force_params_hist + force_params_hist_

            dt_hist = dt_hist + list(jnp.ones(number_of_simulations) * dt)
            iterations_hist = iterations_hist + list(jnp.ones(number_of_simulations) * iterations)
            damping_hist = damping_hist + list(jnp.ones(number_of_simulations) * damping)

        df = pd.DataFrame()
        # save loss and metrics to csv
        df['loss'] = loss_hist
        df['auc'] = [metrics.auc for metrics in metrics_hist]
        df['f1_binary'] = [metrics.f1_binary for metrics in metrics_hist]
        df['f1_micro'] = [metrics.f1_micro for metrics in metrics_hist]
        df['f1_macro'] = [metrics.f1_macro for metrics in metrics_hist]
        df['dt'] = dt_hist
        df['damping'] = damping_hist
        df['iterations'] = iterations_hist
        df.to_csv(f"{OUTPUT_PATH}{'neural' if use_neural_froce else 'spring'}_training_results_{EMBEDDING_DIM}.csv")

    shot_metrics = []
    times = []
    key_shots = random.split(key_test, TEST_SHOTS)     
    graph = {}

    for shot in range(TEST_SHOTS):

        graph = g.to_SignedGraph(dataset, convert_to_undirected)

        training_signs = graph.sign.copy()
        training_signs = jnp.where(graph.train_mask, training_signs, 0)
        training_graph = graph._replace(sign=training_signs)
        training_signs_one_hot = jax.nn.one_hot(training_signs + 1, 3)
        training_graph = training_graph._replace(sign_one_hot=training_signs_one_hot)

        start_time = timer()
        print(f"Running shot {shot + 1} of {TEST_SHOTS}")
   
        # initialize spring state
        node_state = sm.init_node_state(
            rng=key_shots[shot],
            n=graph.num_nodes,
            m=graph.num_edges,
            range=INIT_POS_RANGE,
            embedding_dim=EMBEDDING_DIM
        )

        simulation_params_test = sm.SimulationParams(
            iterations=schedule_params['test_iterations'],
            dt=schedule_params['test_dt'],
            damping=schedule_params['test_damping'])
        
        node_state = sm.simulate(
            simulation_params=simulation_params_test,
            node_state=node_state, 
            use_neural_force=use_neural_froce,
            force_params=force_params,
            graph=training_graph)
        
        end_time = timer()
        print(f"Shot {shot + 1} took {end_time - start_time} seconds")
        times.append(end_time - start_time)

        metrics, pred = sm.evaluate(
            node_state,
            graph.edge_index,
            graph.sign,
            graph.train_mask,
            graph.test_mask)
        
        pred_mu = sm.predict(node_state, graph, 0)
        pred_mu = pred_mu.at[graph.test_mask].get()
        pred_mu = jnp.where(pred_mu > 0.5, 1, -1)

        
        print(f"log regr pred vs mu pred diff: {jnp.mean(jnp.abs(pred - pred_mu))}")
        print(pred)
        print(pred_mu)
    
        
        shot_metrics.append(metrics)

    print(f"average metrics over {TEST_SHOTS} shots:")
    print(f"auc: {np.mean([metrics.auc for metrics in shot_metrics])}")
    print(f"f1_binary: {np.mean([metrics.f1_binary for metrics in shot_metrics])}")
    print(f"f1_micro: {np.mean([metrics.f1_micro for metrics in shot_metrics])}")
    print(f"f1_macro: {np.mean([metrics.f1_macro for metrics in shot_metrics])}")
    print(f"std auc: {np.std([metrics.auc for metrics in shot_metrics])}")
    print(f"std f1_binary: {np.std([metrics.f1_binary for metrics in shot_metrics])}")
    print(f"std f1_micro: {np.std([metrics.f1_micro for metrics in shot_metrics])}")
    print(f"std f1_macro: {np.std([metrics.f1_macro for metrics in shot_metrics])}")

    print(f"average time per shot: {np.mean(times)}")

    questions = [
        inquirer.List('save',
            message="Do you want to visualize the accuracy for the forward simulation?",
            choices=['Yes', 'No'],
        ),
    ]
    answers = inquirer.prompt(questions)
    if answers['save'] == 'Yes':
        graph = g.to_SignedGraph(dataset, convert_to_undirected)

        training_signs = graph.sign.copy()
        training_signs = jnp.where(graph.train_mask, training_signs, 0)
        training_graph = graph._replace(sign=training_signs)
        training_signs_one_hot = jax.nn.one_hot(training_signs + 1, 3)
        training_graph = training_graph._replace(sign_one_hot=training_signs_one_hot)

        # initialize spring state
        node_state = sm.init_node_state(
            rng=key_shots[shot],
            n=graph.num_nodes,
            m=graph.num_edges,
            range=INIT_POS_RANGE,
            embedding_dim=EMBEDDING_DIM
        )

        iter_stride = 5
        simulation_params_test = sm.SimulationParams(
            iterations=iter_stride,
            dt=schedule_params['test_dt'],
            damping=schedule_params['test_damping'])
        
        metrics_hist = []
        energy_hist = []
        mean_edge_distance = []
        iterations_hist = []
        for i in range(1, 100):
            iterations_hist.append(i*iter_stride)
            node_state = sm.simulate(
                simulation_params=simulation_params_test,
                node_state=node_state, 
                use_neural_force=use_neural_froce,
                force_params=force_params,
                graph=training_graph)
            
            metrics, pred = sm.evaluate(
                node_state,
                graph.edge_index,
                graph.sign,
                graph.train_mask,
                graph.test_mask)
            
   
            metrics_hist.append(metrics)
            energy = jnp.linalg.norm(node_state.position, axis=1, keepdims=True) ** 2 * 0.5
            energy_hist.append(jnp.mean(jnp.abs(energy)))

            pos_i = node_state.position[graph.edge_index[0]]
            pos_j = node_state.position[graph.edge_index[1]]
            edge_distance = jnp.linalg.norm(pos_i - pos_j, axis=1)
            mean_edge_distance.append(jnp.mean(edge_distance))

        auc = [metrics.auc for metrics in metrics_hist]
        f1_binary = [metrics.f1_binary for metrics in metrics_hist]
        f1_micro = [metrics.f1_micro for metrics in metrics_hist]
        f1_macro = [metrics.f1_macro for metrics in metrics_hist]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(iterations_hist, auc, label='auc', color='#d73027')
        ax1.plot(iterations_hist, f1_binary, label='f1_binary', color='#fc8d59')
        ax1.plot(iterations_hist, f1_micro, label='f1_micro', color='#91bfdb')
        ax1.plot(iterations_hist, f1_macro, label='f1_macro', color='#4575b4')
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('metrics')
        ax1.legend()
       
        ax2.plot(iterations_hist, energy_hist, label='mean energy', color='#d73027')
        ax2.plot(iterations_hist, mean_edge_distance, label='mean edge distance', color='#4575b4')
        ax2.set_xlabel('iterations')
        ax2.set_ylabel('values')
        ax2.legend()
        plt.show()

    questions = [
        inquirer.List('save',
            message="Do you want to visualize the losses for different training parameters?",
            choices=['Yes', 'No'],
        ),
    ]
    answers = inquirer.prompt(questions)

    iterations = np.linspace(50, 200, 3)
    dampings = np.linspace(0.1, 0.08, 3)

    train_params = sm.TrainingParams(
        num_epochs=250,
        learning_rate=0.05,
        batch_size=number_of_subgraphs,
        init_pos_range=INIT_POS_RANGE,
        embedding_dim=EMBEDDING_DIM,
        multi_step=MULTISTPES_GRADIENT)

    colors = ['#d73027', '#fc8d59', '#4575b4', '#984ea3', '#ff7f00']
    if answers['save'] == 'Yes':
        # gridplot
        fig, axs = plt.subplots(4, 4, figsize=(20, 20))
        axs = axs.flatten()
        iters_grid, dampings_grid = np.meshgrid(iterations, dampings)
        iters_grid = iters_grid.flatten()
        dampings_grid = dampings_grid.flatten()
        for index, (iterations, damping) in enumerate(zip(iters_grid, dampings_grid)):
            sim_params = sm.SimulationParams(
                iterations=int(iterations),
                dt=0.5/iterations,
                damping=damping)
            
            graph = g.to_SignedGraph(dataset, convert_to_undirected)

            training_signs = graph.sign.copy()
            training_signs = jnp.where(graph.train_mask, training_signs, 0)
            training_graph = graph._replace(sign=training_signs)
            training_signs_one_hot = jax.nn.one_hot(training_signs + 1, 3)
            training_graph = training_graph._replace(sign_one_hot=training_signs_one_hot)

            # initialize spring state
            node_state = sm.init_node_state(
                rng=key_shots[shot],
                n=graph.num_nodes,
                m=graph.num_edges,
                range=INIT_POS_RANGE,
                embedding_dim=EMBEDDING_DIM
            )
            
            force_params = sm.init_neural_params()
    
            force_params, loss_hist_, metrics_hist_, force_params_hist_ = sm.train(
                random_key=key_training,
                batches=batches,
                use_neural_force=True,
                force_params=force_params,
                training_params=train_params,
                simulation_params=sim_params)
            
            axs[index].plot(
                loss_hist_, 
                label='d=' + str(damping) + 
                ', iter=' + str(iterations))
            
            axs[index].set_xlabel('epochs')
            axs[index].set_ylabel('loss')
            axs[index].legend()

        plt.show()
        
    questions = [
        inquirer.List('save',
            message="Do you want to plot the embeddings?",
            choices=['Yes', 'No'],
        ),
    ]
    answers = inquirer.prompt(questions)
    if answers['save'] == 'Yes':
        # create new axis
        fig, ax = plt.subplots(1, 1)
        plot_embedding(
            node_state,
            graph,
            ax)
        plt.show()

    if train_parameters:
        questions = [
            inquirer.List('save',
                message="Do you want to save the force parameters?",
                choices=['Yes', 'No'],
            ),
        ]
        answers = inquirer.prompt(questions)
        if answers['save'] == 'Yes':
            with open(force_params_path, 'w') as file:
                yaml.dump(force_params, file)   

    
if __name__ == "__main__":
    main(sys.argv[1:])
    