import jax.numpy as jnp
import jax
from typing import NamedTuple, Optional
from functools import partial
from springs import f1_macro
from gnn import AttentionHead

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

@partial(jax.jit)
def compute_force(
    params : SpringParams, 
    forces_nn_params : jnp.ndarray,
    position_i : jnp.ndarray,
    position_j : jnp.ndarray,
    sign : jnp.ndarray) -> jnp.ndarray:
    
    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + 0.001)

    attraction = jnp.maximum(distance - params.friend_distance, 0) * params.friend_stiffness * spring_vector_norm
    neutral = (distance - NEUTRAL_DISTANCE) * NEUTRAL_STIFFNESS * spring_vector_norm
    retraction = -jnp.maximum(params.enemy_distance - distance, 0) * params.enemy_stiffness * spring_vector_norm

    sign = jnp.expand_dims(sign, axis=1)
    force = jnp.where(sign == 1, attraction, retraction)
    force = jnp.where(sign == 0, neutral, force)
    
    return force

@partial(jax.jit, staticmethods=["dt", "damping", "attention_head"])
def update(
    spring_state : SpringState, 
    spring_params : SpringParams, 
    forces_nn_params : jnp.ndarray,
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

    edge_forces = compute_force(
        spring_state,
        spring_params, 
        position_i, 
        position_j, 
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