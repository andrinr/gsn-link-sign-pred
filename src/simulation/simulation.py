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
    """
    Complete forward integration of the simulation using the Euler method.
    """
    
    # capture the spring_params and signs in the closure
    simulation_update = lambda i, node_state: sm.euler_step(
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

    training_signs = graph.sign.copy()
    training_signs = jnp.where(graph.train_mask, training_signs, 0)
    training_graph = graph._replace(sign=training_signs)

    training_signs_one_hot = jax.nn.one_hot(training_signs + 1, 3)
    training_graph = training_graph._replace(sign_one_hot=training_signs_one_hot)

    node_state = simulate(
        simulation_params = simulation_params,
        node_state = node_state,
        use_neural_force = use_neural_force,
        force_params = force_params,
        graph=training_graph)

    # # We evalute the loss function for different threeshold values to approximate the behavior of the auc metric
    # x_0s = jnp.linspace(-1.0, 1.0, 10)
    # losses = jnp.array([loss(node_state, graph, x_0, simulation_params.threshold) for x_0 in x_0s])

    # loss_value = jnp.mean(losses)

    loss_value = loss(node_state, graph, 0, simulation_params.threshold)    
    
    predicted_sign = predict(
        node_state = node_state,
        graph = graph,
        x_0 = 0,
        threshold = simulation_params.threshold)

    return loss_value, (node_state, predicted_sign)

def predict(
    node_state : sm.NodeState,
    graph : g.SignedGraph,
    x_0 : float,
    threshold : float,
) -> jnp.ndarray:
    
    position_i = node_state.position[graph.edge_index[0]]
    position_j = node_state.position[graph.edge_index[1]]

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
    """
    Differentiable loss function for link sign prediction.
    """
    predicted_sign = predict(
        node_state = node_state,
        graph = graph,
        x_0 = x_0,
        threshold = threshold)

    sign_binary = graph.sign * 0.5 + 0.5

    incorrect_predictions = jnp.square(sign_binary - predicted_sign)

    fraction_negatives = jnp.sum(sign_binary == 0) / sign_binary.shape[0]
    fraction_positives =  1 - fraction_negatives

    weight_positives = 1 / fraction_positives
    weight_negatives = 1 / fraction_negatives

    # score = (sign * predicted_sign) *  1 / fraction_positives + ((1 - sign) * (1 - predicted_sign)) * 1 / fraction_negatives
    
    # false negatives
    jnp.where(sign_binary == 1, incorrect_predictions * weight_positives, incorrect_predictions * weight_negatives)
    
    incorrect_predictions_weighted = jnp.where(
        sign_binary == 1, 
        incorrect_predictions * weight_positives, 
        incorrect_predictions * weight_negatives)
    
    incorrect_predictions_weighted *= jnp.where(graph.test_mask, 0.8, 0.2)

    loss = jnp.mean(incorrect_predictions_weighted)

    return loss