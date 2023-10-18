import jax.numpy as jnp
import jax
from typing import NamedTuple, Optional
from functools import partial
from simulation import f1_macro
from neural import mlp

NEUTRAL_DISTANCE = 10.0
NEUTRAL_STIFFNESS = 10.0

class SpringParams(NamedTuple):
    friend_distance: float
    friend_stiffness: float
    enemy_distance: float
    enemy_stiffness: float

class SpringState(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray
    auxillaries: jnp.ndarray

def init_spring_state(rng : jax.random.PRNGKey, n : int, m : int, embedding_dim : int) -> SpringState:
    position = jax.random.uniform(rng, (n, embedding_dim), maxval=1.0, minval=-1.0)
    velocity = jnp.zeros((n, embedding_dim))
    auxillaries = jnp.random.uniform(rng, (m, embedding_dim), maxval=1.0, minval=-1.0)

    return SpringState(position, velocity, auxillaries)

@partial(jax.jit, static_argnames=["nn_based_forces"])
def compute_force(
    params : SpringParams, 
    nn_based_forces : bool,
    forces_nn_params : dict[jnp.ndarray],
    position_i : jnp.ndarray,
    position_j : jnp.ndarray,
    auxillaries_i : jnp.ndarray,
    auxillaries_j : jnp.ndarray,
    sign : jnp.ndarray) -> jnp.ndarray:

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + 0.001)

    # neural network based forces
    if nn_based_forces:
        forces = mlp(
            jnp.concatenate([spring_vector, auxillaries_i, auxillaries_j, sign], axis=-1),
            forces_nn_params)
    
    # social balance theory based forces
    else:
        attraction = jnp.maximum(distance - params.friend_distance, 0) * params.friend_stiffness
        neutral = (distance - NEUTRAL_DISTANCE) * NEUTRAL_STIFFNESS
        retraction = -jnp.maximum(params.enemy_distance - distance, 0) * params.enemy_stiffness

        sign = jnp.expand_dims(sign, axis=1)
        forces = jnp.where(sign == 1, attraction, retraction)
        forces = jnp.where(sign == 0, neutral, forces)
    
    return forces * spring_vector_norm

@partial(jax.jit, staticmethods=["dt", "damping", "attention_head"])
def update_spring_state(
    spring_state : SpringState, 
    spring_params : SpringParams, 
    nn_based_forces : bool,
    forces_nn_params : dict[jnp.ndarray],
    dt : float,
    damping : float,
    edge_index : jnp.ndarray,
    sign : jnp.ndarray) -> SpringState:
    """
    Update the spring state using the leapfrog method. 
    This is essentially a simple message passing network implementation. 
    """

    position_i = spring_state.position[edge_index[0]]
    position_j = spring_state.position[edge_index[1]]

    auxillaries_i = spring_state.auxillaries[edge_index[0]]
    auxillaries_j = spring_state.auxillaries[edge_index[1]]

    edge_forces = compute_force(
        spring_state,
        spring_params, 
        nn_based_forces,
        forces_nn_params,
        position_i, 
        position_j, 
        auxillaries_i,
        auxillaries_j,
        sign)
    
    node_forces = jnp.zeros_like(spring_state.position)
    node_forces = node_forces.at[edge_index[0]].add(edge_forces)

    velocity = spring_state.velocity + 0.5 * dt * node_forces
    position = spring_state.position + dt * velocity

    edge_forces = compute_force(spring_params, position_i, position_j, sign)
    node_forces = jnp.zeros_like(spring_state.position)
    node_forces = node_forces.at[edge_index[0]].add(edge_forces)

    velocity = velocity + 0.5 * dt * node_forces

    velocity = velocity * (1.0 - damping)

    spring_state = spring_state._replace(
        position=position,
        velocity=velocity)
    
    return spring_state