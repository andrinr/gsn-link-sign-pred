import jax.numpy as jnp
import jax
from typing import NamedTuple, Optional
from functools import partial
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
    auxillary: jnp.ndarray

def init_spring_state(
    rng : jax.random.PRNGKey, 
    n : int, m : int,
    embedding_dim : int,
    auxillary_dim : int) -> SpringState:
    position = jax.random.uniform(rng, (n, embedding_dim), maxval=1.0, minval=-1.0)
    velocity = jnp.zeros((n, embedding_dim))
    auxillary = jax.random.uniform(rng, (n, auxillary_dim), maxval=1.0, minval=-1.0)

    return SpringState(position, velocity, auxillary)

@partial(jax.jit, static_argnames=["nn_force", "nn_auxillary", "params"])
def compute_force(
    state : SpringState,
    params : SpringParams, 
    nn_auxillary : bool,
    nn_force : bool,
    nn_force_params : dict[jnp.ndarray],
    edge_index : jnp.ndarray,
    sign : jnp.ndarray) -> jnp.ndarray:

    position_i = state.position[edge_index[0]]
    position_j = state.position[edge_index[1]]

    auxillaries_i = state.auxillary[edge_index[0]]
    auxillaries_j = state.auxillary[edge_index[1]]

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    difference = distance - NEUTRAL_DISTANCE
    spring_vector_norm = spring_vector / (distance + 0.001)

    # = jnp.expand_dims(sign, axis=1)
    sign_one_hot = jax.nn.one_hot(sign, 3)

    # neural network based forces
    if nn_force and nn_auxillary:
        forces = mlp(
            jnp.concatenate([auxillaries_i, auxillaries_j, difference, sign_one_hot], axis=-1),
            nn_force_params) * 10

    elif nn_force:
        forces = mlp(
            jnp.concatenate([difference, sign_one_hot], axis=-1),
            nn_force_params) * 10
    
    # social balance theory based forces
    else:
        attraction = jnp.maximum(distance - params.friend_distance, 0) * params.friend_stiffness
        neutral = (distance - NEUTRAL_DISTANCE) * NEUTRAL_STIFFNESS
        retraction = -jnp.maximum(params.enemy_distance - distance, 0) * params.enemy_stiffness

        forces = jnp.where(sign == 1, attraction, retraction)
        forces = jnp.where(sign == 0, neutral, forces)
    
    return forces * spring_vector_norm

@partial(jax.jit, static_argnames=["dt", "damping", "nn_force", "nn_auxillary", "spring_params"])
def update_spring_state(
    spring_state : SpringState, 
    spring_params : SpringParams, 
    nn_auxillary : bool,
    nn_force : bool,
    nn_force_params : dict[jnp.ndarray],
    dt : float,
    damping : float,
    edge_index : jnp.ndarray,
    sign : jnp.ndarray) -> SpringState:
    """
    Update the spring state using the leapfrog method. 
    This is essentially a simple message passing network implementation. 
    """
    edge_forces = compute_force(
        spring_state,
        spring_params, 
        nn_auxillary,
        nn_force,
        nn_force_params,
        edge_index,
        sign)
    
    node_forces = jnp.zeros_like(spring_state.position)
    node_forces = node_forces.at[edge_index[0]].add(edge_forces)

    velocity = spring_state.velocity + 0.5 * dt * node_forces
    position = spring_state.position + dt * velocity

    spring_state = spring_state._replace(
        position=position,
        velocity=velocity)

    edge_forces = compute_force(
        spring_state,
        spring_params, 
        nn_auxillary,
        nn_force,
        nn_force_params,
        edge_index,
        sign)
    
    node_forces = jnp.zeros_like(spring_state.position)
    node_forces = node_forces.at[edge_index[0]].add(edge_forces)

    velocity = velocity + 0.5 * dt * node_forces
    velocity = velocity * (1.0 - damping)

    spring_state = spring_state._replace(
        position=position,
        velocity=velocity)
    
    return spring_state