import jax.numpy as jnp
import jax
from functools import partial

import graph as g
import simulation as sm

@partial(jax.jit, static_argnames=["simulation_params", "use_neural_force"])
def simulate(
    simulation_params : sm.SimulationParams,
    spring_state : sm.SpringState,
    force_params : sm.HeuristicForceParams | sm.NeuralForceParams,
    use_neural_force : bool,
    graph : g.SignedGraph
) -> sm.SpringState:
    
    # capture the spring_params and signs in the closure
    simulation_update = lambda i, state: sm.update_spring_state(
        simulation_params = simulation_params, 
        force_params = force_params,
        use_neural_force = use_neural_force,
        spring_state = state,
        graph = graph)

    spring_state = jax.lax.fori_loop(
        0, 
        simulation_params.iterations, 
        simulation_update,
        spring_state)
    
    return spring_state

@partial(jax.jit, static_argnames=["simulation_params", "use_neural_force"])
def simulate_and_loss(
    simulation_params : sm.SimulationParams,
    spring_state : sm.SpringState,
    force_params : sm.HeuristicForceParams | sm.NeuralForceParams,
    use_neural_force : bool,
    graph : g.SignedGraph,
) -> sm.SpringState:

    training_signs = graph.sign.copy()
    training_signs = jnp.where(graph.train_mask, training_signs, 0)
    training_graph = graph._replace(sign=training_signs)

    spring_state = simulate(
        simulation_params = simulation_params,
        spring_state = spring_state,
        force_params = force_params,
        use_neural_force = use_neural_force,
        graph=training_graph)

    # We evalute the loss function for different threeshold values to approximate the behavior of the auc metric
    x_0s = jnp.linspace(-2, 2, 10)
    losses = jnp.array([loss(spring_state, graph, x_0) for x_0 in x_0s])

    loss_value = jnp.mean(losses)

    predicted_sign = predict(
        spring_state = spring_state,
        graph = graph,
        x_0 = 0)

    return loss_value, (spring_state, predicted_sign)

def predict(
    spring_state : sm.SpringState,
    graph : g.SignedGraph,
    x_0 : float
):
    position_i = spring_state.position[graph.edge_index[0]]
    position_j = spring_state.position[graph.edge_index[1]]

    distance = jnp.linalg.norm(position_j - position_i, axis=1) - 10

    # apply sigmoid function to get sign (0 for negative, 1 for positive)
    predicted_sign = 1 / (1 + jnp.exp(1 * (distance-x_0)))

    return predicted_sign

def loss(
    spring_state : sm.SpringState,
    graph : g.SignedGraph,
    x_0 : float
):
    predicted_sign = predict(
        spring_state = spring_state,
        graph = graph,
        x_0 = x_0)

    sign = graph.sign * 0.5 + 0.5

    incorrect_predictions = jnp.square(sign - predicted_sign)

    fraction_negatives = jnp.sum(sign == 0) / sign.shape[0]
    fraction_positives =  1 - fraction_negatives

    weight_positives = 1 / fraction_positives
    weight_negatives = 1 / fraction_negatives

    # score = (sign * predicted_sign) *  1 / fraction_positives + ((1 - sign) * (1 - predicted_sign)) * 1 / fraction_negatives
    
    # false negatives
    jnp.where(sign == 1, incorrect_predictions * weight_positives, incorrect_predictions * weight_negatives)
    
    incorrect_predictions_weighted = jnp.where(
        sign == 1, 
        incorrect_predictions * weight_positives, 
        incorrect_predictions * weight_negatives)

    loss = jnp.mean(incorrect_predictions_weighted)

    return loss