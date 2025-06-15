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
from typing import NamedTuple

# Local dependencies
import simulation as sm
import graph as g
from io_helpers import get_dataset
import stats as stats
from plot_helpers import plot_embedding

class TestParameters(NamedTuple):
    use_blackbox: bool
    num_dimensions: int
    num_simulations: int
    test_iterations: int
    test_dt: float
    test_damping: float
    init_pos_range: float
    use_neural_force: bool
    threshold: float
    num_shots: int

def main(argv) -> None:
    """
    Main function
    """
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".95"

    # Set the precision to 64 bit in case of NaN gradients
    jax.config.update("jax_enable_x64", False)
    print(xla_bridge.get_backend().platform)

    # Paths
    DATA_PATH = 'src/data/'
    MODEL_PATH = 'models/'
    OUTPUT_PATH = 'plots/data/'

    dataset_name = argv[0]
    params_path = argv[1]

    # Load parameters
    with open(params_path, 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    params = TestParameters(**params)
    # Create initial values for neural network parameters
    key_params, key_training, key_test = random.split(random.PRNGKey(2), 3)

    dataset = get_dataset(DATA_PATH, dataset_name)

    dataset = g.to_SignedGraph(dataset, True)
        
    if params.use_neural_force:
        force_params_path = f"{MODEL_PATH}SPRING-NN.yaml"
        with open(force_params_path, 'r') as file:
            force_params = yaml.load(file, Loader=yaml.FullLoader)
    else:
        force_params_path = f"{MODEL_PATH}SPRING.yaml"
        with open(force_params_path, 'r') as file:
            force_params = yaml.load(file, Loader=yaml.FullLoader)


    graph = g.to_SignedGraph(dataset, True)

    training_signs = graph.sign.copy()
    training_signs = jnp.where(graph.train_mask, training_signs, 0)
    training_graph = graph._replace(sign=training_signs)
    training_signs_one_hot = jax.nn.one_hot(training_signs + 1, 3)
    training_graph = training_graph._replace(sign_one_hot=training_signs_one_hot)


    # initialize spring state
    node_state = sm.init_node_state(
        rng=key_training,
        n=graph.num_nodes,
        m=graph.num_edges,
        range=params.init_pos_range,
        embedding_dim=params.embedding_dim
    )

    iter_stride = 2
    simulation_params_test = sm.SimulationParams(
        iterations=iter_stride,
        dt=params.test_dt,
        damping=params.damping)
    
    metrics_hist = []
    vel_hist = []
    acc_hist = []
    iterations_hist = []
    for i in range(1, 100):
        iterations_hist.append(i*iter_stride)
        node_state = sm.simulate(
            simulation_params=simulation_params_test,
            node_state=node_state, 
            use_neural_force=params.use_neural_force,
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
    df.to_csv(f"{OUTPUT_PATH}{'neural' if params.use_neural_froce else 'spring'}_forward_euler_results_{embedding_dim}.csv")

    
if __name__ == "__main__":
    main(sys.argv[1:])
    