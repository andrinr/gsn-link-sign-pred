import jax.numpy as jnp
import jax
from typing import NamedTuple
from functools import partial
from simulation import f1_macro
import simulation as sim

class SimulationParams(NamedTuple):
    iterations : int
    dt : float
    damping : float
    message_passing_iterations : int

@partial(jax.jit, static_argnames=["simulation_params", "nn_force", "nn_auxillary", "spring_params"])
def simulate(
    simulation_params : SimulationParams,
    spring_state : sim.SpringState,
    spring_params : sim.SpringParams,
    nn_auxillary : bool,
    nn_auxillary_params : dict,
    nn_force : bool,
    nn_force_params : dict,
    edge_index : jnp.ndarray,
    signs : jnp.ndarray) -> sim.SpringState:

    if nn_auxillary:
        # capture the auxillaries_nn_params in the closure
        auxillary_update = lambda i, state: sim.update_auxillary_state(
            spring_state = state,
            auxillaries_nn_params = nn_auxillary_params,
            edge_index = edge_index,
            sign = signs)
        
        # jax.debug.print(f"message_passing_iterations: {simulation_params.message_passing_iterations}") 
        spring_state = jax.lax.fori_loop(
            0,
            simulation_params.message_passing_iterations,
            auxillary_update,
            spring_state)
        
    # print(f"spring_state.auxillaries: {spring_state.auxillaries}")

    # capture the spring_params and signs in the closure
    simulation_update = lambda i, state: sim.update_spring_state(
        spring_state = state,
        spring_params = spring_params,
        nn_auxillary = nn_auxillary,
        nn_force = nn_force,
        nn_force_params = nn_force_params,
        dt = simulation_params.dt,
        damping = simulation_params.damping,
        edge_index = edge_index,
        sign = signs)

    spring_state = jax.lax.fori_loop(
        0, 
        simulation_params.iterations, 
        simulation_update,
        spring_state)
    
    return spring_state

@partial(jax.jit, static_argnames=["simulation_params", "nn_force", "nn_auxillary", "spring_params"])
def simulate_and_loss(
    simulation_params : SimulationParams,
    spring_state : sim.SpringState,
    spring_params : sim.SpringParams,
    nn_auxillary : bool,
    nn_auxillary_params : dict,
    nn_force : bool,
    nn_force_params : dict,
    edge_index : jnp.ndarray,
    signs : jnp.ndarray,
    training_mask : jnp.ndarray,
    validation_mask : jnp.ndarray) -> sim.SpringState:

    training_signs = signs.copy()
    training_signs = jnp.where(training_mask, training_signs, 0)

    spring_state = simulate(
        simulation_params = simulation_params,
        spring_state = spring_state,
        spring_params = spring_params,
        nn_auxillary = nn_auxillary,
        nn_auxillary_params = nn_auxillary_params,
        nn_force = nn_force,
        nn_force_params = nn_force_params,
        edge_index = edge_index,
        signs = training_signs)

    position_i = spring_state.position[edge_index[0]]
    position_j = spring_state.position[edge_index[1]]

    distance = jnp.linalg.norm(position_i - position_j, axis=1) - sim.NEUTRAL_DISTANCE

    # apply sigmoid function to get sign (0 for negative, 1 for positive)
    predicted_sign = 1 / (1 + jnp.exp(-distance))

    # apply same transformation to the actual signs
    signs = signs * 0.5 + 0.5

    incorrect_predictions = (predicted_sign - signs) ** 2

    fraction_negatives = jnp.sum(signs == 0) / signs.shape[0]
    fraction_positives =  1 - fraction_negatives

    loss = jnp.sum(jnp.where(signs == 1, incorrect_predictions * 1 / fraction_positives, incorrect_predictions * 1 / fraction_negatives))
    
    # apply weights to the loss
    # incorrect_predictions = jnp.where(signs == 1, incorrect_predictions * weight_positives, incorrect_predictions * weight_negatives)

    # loss = -f1_macro(signs, predicted_sign)
    
    return loss, (spring_state, predicted_sign)
