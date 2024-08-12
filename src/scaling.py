# External dependencies
import sys
import torch_geometric.datasets
import torch_geometric.datasets.graph_generator
import yaml
from jax import random
from jax.lib import xla_bridge
import jax.numpy as jnp
import os
import jax
import os
import sys
from typing import NamedTuple
from IPython import get_ipython
import torch
import torch_geometric
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

    params_path = argv[0]

    # Load parameters
    with open(params_path, 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    params = TestParameters(**params)
    # Create initial values for neural network parameters
    key_params, key_training, key_test = random.split(random.PRNGKey(2), 3)
        
    if params.use_neural_force:
        force_params_path = f"{MODEL_PATH}SPRING-NN.yaml"
        with open(force_params_path, 'r') as file:
            force_params = yaml.load(file, Loader=yaml.UnsafeLoader)
    else:
        force_params_path = f"{MODEL_PATH}SPRING.yaml"
        with open(force_params_path, 'r') as file:
            force_params = yaml.load(file, Loader=yaml.UnsafeLoader)
    
    for num_edges in [100, 1000, 10000, 100000]:
        num_nodes = 1000
        data = torch_geometric.data.Data()
        data.edge_index = torch.randint(0, num_nodes, (2, num_edges))
        num_edges = data.edge_index.shape[1]
        print(f"Number of nodes: {num_nodes}, number of edges: {num_edges}")
        signs = torch.randint(0, 2, (num_edges,))
        data.edge_attr = torch.where(signs == 0, -1, 1)
        graph = g.to_SignedGraph(data, True)

        training_signs = graph.sign.copy()
        training_signs = jnp.where(graph.train_mask, training_signs, 0)
        training_graph = graph._replace(sign=training_signs)
        training_signs_one_hot = jax.nn.one_hot(training_signs + 1, 3)
        training_graph = training_graph._replace(sign_one_hot=training_signs_one_hot)

        sim_jit = jax.jit(sm.simulate, static_argnames=["simulation_params", "use_neural_force"])

        simulation_params_test = sm.SimulationParams(
            iterations=params.test_iterations,
            dt=params.test_dt,
            damping=params.test_damping,
            threshold=params.threshold)

        node_state = sm.init_node_state(
            rng=key_test,
            n=graph.num_nodes,
            m=graph.num_edges,
            range=params.init_pos_range,
            embedding_dim=params.num_dimensions)

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
    
    # # disable JIT compilation
    # jax.config.update("jax_disable_jit", True)
    # sim_no_jit = sm.simulate

    # print("Execution time without JIT compilation:")
    # ipython.run_line_magic("timeit", 'sim_no_jit(' +\
    #     'simulation_params=simulation_params_test,' +\
    #     'node_state=node_state,' +\
    #     'use_neural_force=params.use_neural_force, ' +\
    #     'force_params=force_params, ' +\
    #     'graph=training_graph)')


if __name__ == "__main__":
    main(sys.argv[1:])
    