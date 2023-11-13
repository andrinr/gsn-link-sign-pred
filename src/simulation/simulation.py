import jax.numpy as jnp
import jax
from functools import partial
import simulation as sim
from graph import SignedGraph

# @partial(jax.jit, static_argnames=["simulation_params", "nn_force", "nn_auxillary"])
def simulate(
    simulation_params : sim.SimulationParams,
    spring_state : sim.SpringState,
    spring_params : sim.SpringParams,
    nn_force : bool,
    nn_auxillary_params : dict,
    nn_force_params : dict,
    graph : SignedGraph) -> sim.SpringState:

    if nn_force:
        # capture the auxillaries_nn_params in the closure
        auxillary_update = lambda i, state: sim.update_auxillary_state(
            spring_state = state,
            auxillaries_nn_params = nn_auxillary_params,
            graph = graph)
        
        # jax.debug.print(f"message_passing_iterations: {simulation_params.message_passing_iterations}") 
        spring_state = jax.lax.fori_loop(
            0,
            simulation_params.message_passing_iterations,
            auxillary_update,
            spring_state)
        
    spring_state = sim.force_decision(
        spring_state=spring_state,
        nn_force=nn_force,
        nn_force_params=nn_force_params,
        graph = graph)
    
    # capture the spring_params and signs in the closure
    simulation_update = lambda i, state: sim.update_spring_state(
        simulation_params = simulation_params, 
        spring_params = spring_params,
        spring_state = state,
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
    predicted_sign = 1 / (1 + jnp.exp(distance))

    return predicted_sign

@partial(jax.jit, static_argnames=["simulation_params", "nn_force"])
def simulate_and_loss(
    simulation_params : sim.SimulationParams,
    spring_state : sim.SpringState,
    spring_params : sim.SpringParams,
    nn_force : bool,
    nn_auxillary_params : dict,
    nn_force_params : dict,
    graph : SignedGraph) -> sim.SpringState:

    training_signs = graph.sign.copy()
    training_signs = jnp.where(graph.train_mask, training_signs, 0)
    training_graph = graph._replace(sign=training_signs)

    spring_state = simulate(
        simulation_params = simulation_params,
        spring_state = spring_state,
        spring_params = spring_params,
        nn_force = nn_force,
        nn_auxillary_params = nn_auxillary_params,
        nn_force_params = nn_force_params,
        graph=training_graph)

    predicted_sign = predict(
        spring_state = spring_state,
        spring_params = spring_params,
        graph = graph)

    # apply same transformation to the actual signs
    sign = graph.sign * 0.5 + 0.5

    incorrect_predictions = (sign - predicted_sign) ** 2

    fraction_negatives = jnp.sum(sign == 0) / sign.shape[0]
    fraction_positives =  1 - fraction_negatives

    weight_positives = 1 / fraction_positives
    weight_negatives = 1 / fraction_negatives

    # score = (sign * predicted_sign) *  1 / fraction_positives + ((1 - sign) * (1 - predicted_sign)) * 1 / fraction_negatives
    
    # apply weights to the loss
    incorrect_predictions = jnp.where(sign == 1, incorrect_predictions * weight_positives, incorrect_predictions * weight_negatives)
    # only consider the training / validation nodes
    # incorrect_predictions = jnp.where(graph.test_mask, 0, incorrect_predictions)

    # MSE loss
    loss = jnp.mean(incorrect_predictions)
    
    return loss, (spring_state, predicted_sign)
