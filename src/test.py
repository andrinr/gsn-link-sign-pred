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
import os
import argparse, sys

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
    # Deactivate preallocation of memory to avoid OOM errors
    #os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".95"
    #os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

    # Set the precision to 64 bit in case of NaN gradients
    jax.config.update("jax_enable_x64", True)
    print(xla_bridge.get_backend().platform)
    # disable jit compilation for debugging
    jax.config.update("jax_disable_jit", False)

    # Paths
    DATA_PATH = 'src/data/'
    CECKPOINT_PATH = 'training_checkpoints/'
    OUTPUT_PATH = 'plots/data/'
    
    # set data path
    dataset_name = argv[0]

    params = {
        "num_dimensions": 32,
        "num_simulations": 100,
        "train_iterations": 100,
        "train_dt": 0.01,
        "train_damping": 0.05,
        "learning_rate": 0.03,
        "init_pos_range": 1.0,
        "enable_partitioning": False,
        "number_of_subgraphs": 4,
        "use_neural_force": False,
        "gradient_multisteps": 1,
    }

    # Create initial values for neural network parameters
    key_params, key_training, key_test = random.split(random.PRNGKey(2), 3)

    dataset = get_dataset(DATA_PATH, dataset_name)

    batches = []
    if params.enable_partitioning:
        cluster_data = ClusterData(
            dataset, 
            num_parts=params.number_of_subgraphs,
            keep_inter_cluster_edges=True)
        
        loader = ClusterLoader(cluster_data)
        for batch in loader:
            signedGraph = g.to_SignedGraph(batch, True)
            print(f"num_nodes: {signedGraph.num_nodes}")
            if signedGraph.num_nodes > 50:
                batches.append(signedGraph)
    else:
        batches.append(g.to_SignedGraph(dataset, True))
        
    if params.use_neural_force:
        force_params = sm.init_neural_params(key_params)
    else:
        force_params = sm.init_spring_force_params()

    loss_hist = []
    metrics_hist = []
    force_params_hist = []
    dt_hist = []
    damping_hist = []
    iterations_hist = []

    number_of_simulations = params.num_simulations
    iterations = params.train_iterations
    dt = params.train_dt
    damping = params.train_damping

    simulation_params_train = sm.SimulationParams(iterations=iterations, dt=dt, damping=damping)

    print(f"Running {number_of_simulations} simulations with {iterations} iterations each")
    print(f"Simulation parameters are dt: {dt}, damping: {damping}")

    force_params, loss_hist_, metrics_hist_, force_params_hist_ = sm.gradient_training(
        random_key=key_training,
        batches=batches,
        use_neural_force=params.use_neural_force,
        force_params=force_params,
        training_params= sm.TrainingParams(
            num_epochs=number_of_simulations,
            learning_rate=params.learning_rate,
            batch_size=params.number_of_subgraphs,
            init_pos_range=params.init_pos_range,
            embedding_dim=params.num_dimensions,
            multi_step=params.gradient_multisteps),
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
    df['auc_p'] = [metrics.auc_p for metrics in metrics_hist]
    df['auc_l'] = [metrics.auc_l for metrics in metrics_hist]
    df['f1_binary'] = [metrics.f1_binary for metrics in metrics_hist]
    df['f1_micro'] = [metrics.f1_micro for metrics in metrics_hist]
    df['f1_macro'] = [metrics.f1_macro for metrics in metrics_hist]
    df['f1_weighted'] = [metrics.f1_weighted for metrics in metrics_hist]
    df['dt'] = dt_hist
    df['damping'] = damping_hist
    df['iterations'] = iterations_hist
    df.to_csv(f"{OUTPUT_PATH}training_process.csv")

    shot_metrics = []
    times = []
    key_shots = random.split(key_test, params.num)     
    graph = {}

    for i in range(1):
        if i == 0:
            jax.config.update("jax_disable_jit", False)
            print("JIT enabled")
        else:
            jax.config.update("jax_disable_jit", True)
            print("JIT disabled")

        for shot in range(num_test_shots):

            graph = g.to_SignedGraph(dataset, True)

            training_signs = graph.sign.copy()
            training_signs = jnp.where(graph.train_mask, training_signs, 0)
            training_graph = graph._replace(sign=training_signs)
            training_signs_one_hot = jax.nn.one_hot(training_signs + 1, 3)
            training_graph = training_graph._replace(sign_one_hot=training_signs_one_hot)

            start_time = timer()
            print(f"Running shot {shot + 1} of {num_test_shots}")

            # initialize spring state
            node_state = sm.init_node_state(
                rng=key_shots[shot],
                n=graph.num_nodes,
                m=graph.num_edges,
                range=init_pos_range,
                embedding_dim=embedding_dim
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
        
            shot_metrics.append(metrics)

        print(f"average metrics over {num_test_shots} shots:")
        print(f"f1_micro: {np.mean([metrics.f1_micro for metrics in shot_metrics])}")
        print(f"f1_micro std: {np.std([metrics.f1_micro for metrics in shot_metrics])}")
        print(f"f1_macro: {np.mean([metrics.f1_macro for metrics in shot_metrics])}")
        print(f"f1_macro std: {np.std([metrics.f1_macro for metrics in shot_metrics])}")
        print(f"f1_weighted: {np.mean([metrics.f1_weighted for metrics in shot_metrics])}")
        print(f"f1_weighted std: {np.std([metrics.f1_weighted for metrics in shot_metrics])}")
        print(f"f1_binary: {np.mean([metrics.f1_binary for metrics in shot_metrics])}")
        print(f"f1_binary std: {np.std([metrics.f1_binary for metrics in shot_metrics])}")
        print(f"auc_p: {np.mean([metrics.auc_p for metrics in shot_metrics])}")
        print(f"auc_p std: {np.std([metrics.auc_p for metrics in shot_metrics])}")
        print(f"auc_l: {np.mean([metrics.auc_l for metrics in shot_metrics])}")
        print(f"auc_l std: {np.std([metrics.auc_l for metrics in shot_metrics])}")

        # check if file exists
        if not os.path.exists(f"{OUTPUT_PATH}benchmark.csv"):
            df = pd.DataFrame(columns=['auc_p', 'auc_l', 'f1_binary', 'f1_micro', 'f1_macro', 'f1_weighted', 'std_auc_p', 'std_auc_l', 'std_f1_binary', 'std_f1_micro', 'std_f1_macro', 'std_f1_weighted', 'dataset', 'embedding_dim', 'method'])
            df.to_csv(f"{OUTPUT_PATH}benchmark.csv")
        else:
            df = pd.read_csv(f"{OUTPUT_PATH}benchmark.csv")
           
        # add new row to dataframe
        df = df._append({
            'auc_p': np.mean([metrics.auc_p for metrics in shot_metrics]),
            'auc_l': np.mean([metrics.auc_l for metrics in shot_metrics]),
            'f1_binary': np.mean([metrics.f1_binary for metrics in shot_metrics]),
            'f1_micro': np.mean([metrics.f1_micro for metrics in shot_metrics]),
            'f1_macro': np.mean([metrics.f1_macro for metrics in shot_metrics]),
            'f1_weighted': np.mean([metrics.f1_weighted for metrics in shot_metrics]),
            'std_auc_p': np.std([metrics.auc_p for metrics in shot_metrics]),
            'std_auc_l': np.std([metrics.auc_l for metrics in shot_metrics]),
            'std_f1_binary': np.std([metrics.f1_binary for metrics in shot_metrics]),
            'std_f1_micro': np.std([metrics.f1_micro for metrics in shot_metrics]),
            'std_f1_macro': np.std([metrics.f1_macro for metrics in shot_metrics]),
            'std_f1_weighted': np.std([metrics.f1_weighted for metrics in shot_metrics]),
            'dataset': dataset_name,
            'embedding_dim': embedding_dim,
            'method': 'neural' if use_neural_froce else 'spring'
        }, ignore_index=True)

        df.to_csv(f"{OUTPUT_PATH}benchmark.csv")

        print(f"average time per shot: {np.mean(times)}")

    questions = [
        inquirer.List('save',
            message="Output accuraccy over forward euler integration?",
            choices=['Yes', 'No'],
        ),
    ]
    answers = inquirer.prompt(questions)
    if answers['save'] == 'Yes':
        graph = g.to_SignedGraph(dataset, True)

        training_signs = graph.sign.copy()
        training_signs = jnp.where(graph.train_mask, training_signs, 0)
        training_graph = graph._replace(sign=training_signs)
        training_signs_one_hot = jax.nn.one_hot(training_signs + 1, 3)
        training_graph = training_graph._replace(sign_one_hot=training_signs_one_hot)

        # initialize spring state
        node_state = sm.init_node_state(
            rng=key_shots[0],
            n=graph.num_nodes,
            m=graph.num_edges,
            range=init_pos_range,
            embedding_dim=embedding_dim
        )

        iter_stride = 2
        simulation_params_test = sm.SimulationParams(
            iterations=iter_stride,
            dt=schedule_params['test_dt'],
            damping=schedule_params['test_damping'])
        
        metrics_hist = []
        vel_hist = []
        acc_hist = []
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
           
            vel_hist.append(jnp.mean(jnp.linalg.norm(node_state.velocity, axis=1)))
            acc_hist.append(jnp.mean(jnp.linalg.norm(node_state.acceleration, axis=1)))

        auc_l = [metrics.auc_l for metrics in metrics_hist]
        auc_p = [metrics.auc_p for metrics in metrics_hist]
        f1_binary = [metrics.f1_binary for metrics in metrics_hist]
        f1_micro = [metrics.f1_micro for metrics in metrics_hist]
        f1_macro = [metrics.f1_macro for metrics in metrics_hist]
        f1_weighted = [metrics.f1_weighted for metrics in metrics_hist]

        # store metrics to csv
        df = pd.DataFrame()

        df['auc_p'] = auc_p
        df['auc_l'] = auc_l
        df['f1_binary'] = f1_binary
        df['f1_micro'] = f1_micro
        df['f1_macro'] = f1_macro
        df['f1_weighted'] = f1_weighted
        df['mean_velocity'] = vel_hist
        df['mean_acceleration'] = acc_hist
        df['iterations'] = iterations_hist
        df.to_csv(f"{OUTPUT_PATH}{'neural' if use_neural_froce else 'spring'}_forward_euler_results_{embedding_dim}.csv")

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
        init_pos_range=init_pos_range,
        embedding_dim=embedding_dim,
        multi_step=gradient_multisteps)

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
            
            graph = g.to_SignedGraph(dataset, True)

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
                range=init_pos_range,
                embedding_dim=embedding_dim
            )
            
            force_params = sm.init_neural_params()
    
            force_params, loss_hist_, metrics_hist_, force_params_hist_ = sm.gradient_training(
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
    