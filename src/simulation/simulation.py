import jax.numpy as jnp
import jax
from functools import partial
import simulation as sim
from graph import SignedGraph
from optax import softmax_cross_entropy_with_integer_labels

@partial(jax.jit, static_argnames=["simulation_params", "nn_force"])
def simulate(
    simulation_params : sim.SimulationParams,
    spring_state : sim.SpringState,
    spring_params : sim.SpringParams,
    nn_force : bool,
    nn_force_params : dict,
    graph : SignedGraph) -> sim.SpringState:
    
    # capture the spring_params and signs in the closure
    simulation_update = lambda i, state: sim.update_spring_state(
        simulation_params = simulation_params, 
        spring_params = spring_params,
        spring_state = state,
        nn_force = nn_force,
        nn_force_params = nn_force_params,
        graph = graph)

    spring_state = jax.lax.fori_loop(
        0, 
        simulation_params.iterations, 
        simulation_update,
        spring_state)
    
    return spring_state

def predict(
    spring_state : sim.SpringState,
    spring_params : sim.SpringParams,
    graph : SignedGraph
):
    position_i = spring_state.position[graph.edge_index[0]]
    position_j = spring_state.position[graph.edge_index[1]]

    distance = jnp.linalg.norm(position_j - position_i, axis=1) - spring_params.distance_threshold

    # apply sigmoid function to get sign (0 for negative, 1 for positive)
    predicted_sign = 1 / (1 + jnp.exp(1 * distance))

    return predicted_sign

@partial(jax.jit, static_argnames=["simulation_params", "nn_force"])
def simulate_and_loss(
    simulation_params : sim.SimulationParams,
    spring_state : sim.SpringState,
    spring_params : sim.SpringParams,
    nn_force : bool,
    nn_force_params : dict,
    graph : SignedGraph,
    ) -> sim.SpringState:

    training_signs = graph.sign.copy()
    training_signs = jnp.where(graph.train_mask, training_signs, 0)
    training_graph = graph._replace(sign=training_signs)

    spring_state = simulate(
        simulation_params = simulation_params,
        spring_state = spring_state,
        spring_params = spring_params,
        nn_force = nn_force,
        nn_force_params = nn_force_params,
        graph=training_graph)

    predicted_sign = predict(
        spring_state = spring_state,
        spring_params = spring_params,
        graph = graph)
    
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
    
    # we weigh the test data twice as much
    # incorrect_predictions_weighted = jnp.where(graph.test_mask, incorrect_predictions_weighted , 0)

    # MSE loss
    loss = jnp.mean(incorrect_predictions_weighted)
    
    return loss, (spring_state, predicted_sign)
