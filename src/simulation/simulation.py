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

@partial(jax.jit, static_argnames=["simulation_params", "nn_based_forces"])
def simulate(
    simulation_params : SimulationParams,
    spring_state : sim.SpringState,
    spring_params : sim.SpringParams,
    nn_based_forces : bool,
    auxillaries_nn_params : dict[jnp.ndarray],
    forces_nn_params : dict[jnp.ndarray],
    edge_index : jnp.ndarray,
    signs : jnp.ndarray) -> sim.SpringState:

    if nn_based_forces:
        # capture the auxillaries_nn_params in the closure
        auxillary_update = lambda i, state: sim.update_auxillary_state(
            spring_state = state,
            auxillaries_nn_params = auxillaries_nn_params,
            edge_index = edge_index,
            sign = signs)

        spring_state = jax.lax.fori_loop(
            0,
            simulation_params.message_passing_iterations,
            auxillary_update,
            spring_state)

    # capture the spring_params and signs in the closure
    simulation_update = lambda i, state: sim.update_spring_state(
        spring_state = state,
        spring_params = spring_params,
        forces_nn_params = forces_nn_params,
        dt = simulation_params.dt,
        damping = simulation_params.damping,
        edge_index = edge_index,
        signs = signs)

    spring_state = jax.lax.fori_loop(
        0, 
        simulation_params.iterations, 
        simulation_update,
        spring_state)
    
    return spring_state

@partial(jax.jit, static_argnames=["simulation_params", "nn_based_forces"])
def simulate_and_loss(
    simulation_params : SimulationParams,
    spring_state : sim.SpringState,
    spring_params : sim.SpringParams,
    nn_based_forces : bool,
    auxillaries_nn_params : dict[jnp.ndarray],
    forces_nn_params : dict[jnp.ndarray],
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
        nn_based_forces = nn_based_forces,
        auxillaries_nn_params = auxillaries_nn_params,
        forces_nn_params = forces_nn_params,
        edge_index = edge_index,
        signs = training_signs)

    position_i = spring_state.position[edge_index[0]]
    position_j = spring_state.position[edge_index[1]]

    spring_vec_norm = jnp.linalg.norm(position_i - position_j, axis=1)
    
    predicted_sign = spring_vec_norm - sim.NEUTRAL_DISTANCE
    logistic = lambda x: 1 / (1 + jnp.exp(-x))
    predicted_sign = logistic(predicted_sign)

    signs = jnp.where(signs == 1, 1, 0)

    # fraction_negatives = jnp.mean(predicted_sign)
    # fraction_positives = 1 - fraction_negatives

    # score = 1/fraction_positives * signs * predicted_sign + 1/fraction_negatives * (1 - signs) * (1 - predicted_sign)

    # loss = -jnp.mean(score)

    loss = -f1_macro(signs, predicted_sign)
    
    return loss, spring_state
