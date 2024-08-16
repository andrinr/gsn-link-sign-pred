# External dependencies
import sys
from torch_geometric.loader import ClusterData, ClusterLoader
import yaml
from jax import random
from jax.lib import xla_bridge
import jax.numpy as jnp
import os
import jax
import pandas as pd
import os
from typing import NamedTuple

# Local dependencies
import simulation as sm
import graph as g
from io_helpers import get_dataset
import stats as stats
from plot_helpers import plot_embedding

class Parameters(NamedTuple):
    use_blackbox: bool
    num_dimensions: int
    num_simulations: int
    train_iterations: int
    train_dt: float
    train_damping: float
    learning_rate: float
    init_pos_range: float
    enable_partitioning: bool
    number_of_subgraphs: int
    use_neural_force: bool
    gradient_multisteps: int
    threshold: float
    seed : int

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
    MODEL_PATH = 'models/'
    OUTPUT_PATH = 'plots/data/'
    
    # set data path
    dataset_name = argv[0]
    params_path = argv[1]

    # Load parameters
    with open(params_path, 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    params = Parameters(**params)

    # Create initial values for neural network parameters
    key_params, key_training = random.split(random.PRNGKey(params.seed), 2)

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

    simulation_params_train = sm.SimulationParams(
        iterations=params.train_iterations,
        dt=params.train_dt,
        damping=params.train_damping,
        threshold=params.threshold)

    force_params, loss_hist_, metrics_hist_, force_params_hist_ = sm.train(
        random_key=key_training,
        batches=batches,
        use_neural_force=params.use_neural_force,
        force_params=force_params,
        training_params= sm.TrainingParams(
            blackbox=params.use_blackbox,
            num_epochs=params.num_simulations,
            learning_rate=params.learning_rate,
            batch_size=params.number_of_subgraphs,
            init_pos_range=params.init_pos_range,
            embedding_dim=params.num_dimensions,
            multi_step=params.gradient_multisteps),
        simulation_params=simulation_params_train)
    
    loss_hist = loss_hist + loss_hist_
    metrics_hist = metrics_hist + metrics_hist_
    force_params_hist = force_params_hist + force_params_hist_

    dt_hist = dt_hist + list(jnp.ones(params.num_simulations) * params.train_dt)
    iterations_hist = iterations_hist + list(jnp.ones(params.num_simulations) * params.train_iterations)
    damping_hist = damping_hist + list(jnp.ones(params.num_simulations) * params.train_damping)

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

    model_name = f"{MODEL_PATH}{'SPRING-NN' if params.use_neural_force else 'SPRING'}.yaml"
    with open(model_name, 'w') as file:
        yaml.dump(force_params, file)
    
if __name__ == "__main__":
    main(sys.argv[1:])
    