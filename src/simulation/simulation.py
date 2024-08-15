import jax.numpy as jnp
import jax
from functools import partial
from typing import Tuple

import graph as g
import simulation as sm

@partial(jax.jit, static_argnames=["simulation_params", "use_neural_force"])
def simulate(
    simulation_params : sm.SimulationParams,
    node_state : sm.NodeState,
    use_neural_force : bool,
    force_params : sm.NeuralEdgeParams,
    graph : g.SignedGraph
) -> sm.NodeState:
    
    # capture the spring_params and signs in the closure
    simulation_update = lambda i, node_state: sm.update(
        simulation_params = simulation_params, 
        use_neural_force = use_neural_force,
        force_params = force_params,
        node_state = node_state,
        graph = graph)

    node_state = jax.lax.fori_loop(
        0, 
        simulation_params.iterations, 
        simulation_update,
        node_state)
    
    return node_state

@partial(jax.jit, static_argnames=["simulation_params", "use_neural_force"])
def simulate_and_loss(
    simulation_params : sm.SimulationParams,
    node_state : sm.NodeState,
    use_neural_force : bool,
    force_params : sm.NeuralEdgeParams,
    graph : g.SignedGraph,
) -> Tuple[float, Tuple[sm.NodeState, jnp.ndarray]]:

    node_state = simulate(
        simulation_params = simulation_params,
        node_state = node_state,
        use_neural_force = use_neural_force,
        force_params = force_params,
        graph=graph)

    loss_value = loss(node_state, graph, 0, simulation_params.threshold)    
    

    return loss_value, node_state

def predict(
    node_state : sm.NodeState,
    edge_index : jax.Array,
    x_0 : float,
    threshold : float,
) -> jnp.ndarray:
    
    position_i = node_state.position[edge_index[0]]
    position_j = node_state.position[edge_index[1]]

    distance = jnp.linalg.norm(position_j - position_i, axis=1) - threshold

    # apply sigmoid function to get sign (0 for negative, 1 for positive)
    predicted_sign = 1 / (1 + jnp.exp(1 * (distance-x_0)))

    return predicted_sign

def loss(
    node_state : sm.NodeState,
    graph : g.SignedGraph,
    x_0 : float,
    threshold : float,
) -> float:
    predicted_sign_train = predict(
        node_state = node_state,
        edge_index= graph.train_edge_index,
        x_0 = x_0,
        threshold = threshold)
    
    predicted_sign_test = predict(
        node_state = node_state,
        edge_index= graph.test_edge_index,
        x_0 = x_0,
        threshold = threshold)

    sign_binary_train = graph.train_sign * 0.5 + 0.5
    sign_binary_test = graph.test_sign * 0.5 + 0.5

    incorrect_predictions_train = jnp.square(sign_binary_train - predicted_sign_train)
    incorrect_predictions_test = jnp.square(sign_binary_test - predicted_sign_test)

    print(sign_binary_train.shape)
    print(incorrect_predictions_train.shape)

    # fraction_negatives = jnp.sum(sign_binary_train == 0) / sign_binary_train.shape[0]
    # fraction_positives =  1 - fraction_negatives

    # weight_positives = 1 / fraction_positives
    # weight_negatives = 1 / fraction_negatives
    
    loss_train = jnp.mean(jnp.where(
        sign_binary_train == 1, 
        incorrect_predictions_train * 1/0.9, 
        incorrect_predictions_train * 1/0.1))
    
    # loss_test = jnp.mean(jnp.where(
    #     sign_binary_test == 1, 
    #     incorrect_predictions_test * 1/0.9, 
    #     incorrect_predictions_test * 1/0.1))

    return loss_train