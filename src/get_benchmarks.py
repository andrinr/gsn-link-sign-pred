# External dependencies
import sys
import yaml
from jax import random
from jax.lib import xla_bridge
import jax.numpy as jnp
import os
import numpy as np
import jax
import pandas as pd
import os
import sys
from typing import NamedTuple
from IPython import get_ipython
ipython = get_ipython()

# Local dependencies
import simulation as sm
import graph as g
from io_helpers import get_dataset
import stats as stats

class TestParameters(NamedTuple):
    use_blackbox: bool
    num_dimensions: int
    test_iterations: int
    test_dt: float
    test_damping: float
    init_pos_range: float
    use_neural_force: bool
    threshold: float
    num_shots: int
    seed : int

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
    OUTPUT_PATH = 'plots/data/benchmarks/'

    dataset_name = argv[0]
    params_path = argv[1]

    # Load parameters
    with open(params_path, 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    params = TestParameters(**params)
    # Create initial values for neural network parameters
    key = random.PRNGKey(params.seed)

    dataset = get_dataset(DATA_PATH, dataset_name)
        
    if params.use_neural_force:
        force_params_path = f"{MODEL_PATH}SPRING-NN.yaml"
        with open(force_params_path, 'r') as file:
            force_params = yaml.load(file, Loader=yaml.UnsafeLoader)
    else:
        force_params_path = f"{MODEL_PATH}SPRING.yaml"
        with open(force_params_path, 'r') as file:
            force_params = yaml.load(file, Loader=yaml.UnsafeLoader)

    print(force_params)

    shot_metrics = []
    key_shots = random.split(key, params.num_shots)  
    graph = {}

    for shot in range(params.num_shots):
        graph = g.to_SignedGraph(dataset, True)

        training_signs = graph.sign.copy()
        training_signs = jnp.where(graph.train_mask, training_signs, 0)
        training_graph = graph._replace(sign=training_signs)
        training_signs_one_hot = jax.nn.one_hot(training_signs + 1, 3)
        training_graph = training_graph._replace(sign_one_hot=training_signs_one_hot)

        print(f"Running shot {shot + 1} of {params.num_shots}")

        # initialize spring state
        node_state = sm.init_node_state(
            rng=key_shots[shot],
            n=graph.num_nodes,
            m=graph.num_edges,
            range=params.init_pos_range,
            embedding_dim=params.num_dimensions)

        simulation_params_test = sm.SimulationParams(
            iterations=params.test_iterations,
            dt=params.test_dt,
            damping=params.test_damping,
            threshold=params.threshold)
        
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
    
        shot_metrics.append(metrics)

    print(f"average metrics over {params.num_shots} shots:")
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

    df = pd.DataFrame(columns=[
        'auc_p', 'auc_l', 
        'f1_binary', 'f1_micro', 'f1_macro', 'f1_weighted', 
        'std_auc_p', 'std_auc_l', 
        'std_f1_binary', 'std_f1_micro', 'std_f1_macro', 'std_f1_weighted', 
        'dataset', 'embedding_dim', 'method'])

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
        'embedding_dim': params.num_dimensions,
        'method': 'neural' if params.use_neural_force else 'spring'
    }, ignore_index=True)

    df.to_csv(f"{OUTPUT_PATH}{dataset_name}_{params.num_dimensions}_{'nn' if params.use_neural_force else 'spring'}.csv")

    sim_jit = jax.jit(sm.simulate, static_argnames=["simulation_params", "use_neural_force"])

    print("JIT compilation and execution times:")
    # measure compilation time
    ipython.run_line_magic("time", 'jax.block_until_ready(sim_jit(' +\
        'simulation_params=simulation_params_test,' +\
        'node_state=node_state,' +\
        'use_neural_force=params.use_neural_force, ' +\
        'force_params=force_params, ' +\
        'graph=training_graph))')
    
    # measure execution time
    ipython.run_line_magic("timeit", 'jax.block_until_ready(sim_jit(' +\
        'simulation_params=simulation_params_test,' +\
        'node_state=node_state,' +\
        'use_neural_force=params.use_neural_force, ' +\
        'force_params=force_params, ' +\
        'graph=training_graph))')
    
    # disable JIT compilation
    jax.config.update("jax_disable_jit", True)
    sim_no_jit = sm.simulate

    print("Execution time without JIT compilation:")
    ipython.run_line_magic("timeit", 'sim_no_jit(' +\
        'simulation_params=simulation_params_test,' +\
        'node_state=node_state,' +\
        'use_neural_force=params.use_neural_force, ' +\
        'force_params=force_params, ' +\
        'graph=training_graph)')


if __name__ == "__main__":
    main(sys.argv[1:])
    