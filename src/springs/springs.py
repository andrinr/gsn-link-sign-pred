import jax.numpy as jnp
import jax
from typing import NamedTuple
from functools import partial

NEUTRAL_DISTANCE = 1.0
NEUTRAL_STIFFNESS = 1.0

class SpringParams(NamedTuple):
    friend_distance: float
    friend_stiffness: float
    enemy_distance: float
    enemy_stiffness: float
    damping: float
    time_step: float

class SpringState(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray
    energy: jnp.ndarray
    loss : float = 0.0

def init_spring_state(rng : jax.random.PRNGKey, n : int, embedding_dim : int) -> SpringState:
    position = jax.random.uniform(rng, (n, embedding_dim), maxval=1.0, minval=-1.0)
    velocity = jnp.zeros((n, embedding_dim))
    energy = jnp.zeros(n)

    return SpringState(position, velocity, energy)

@partial(jax.jit)
def compute_force(
    params : SpringParams, 
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

@partial(jax.jit)
def update(
    state : SpringState, 
    params : SpringParams, 
    sign : jnp.ndarray, 
    edge_index : jnp.ndarray) -> SpringState:
    """
    Update the spring state using the leapfrog method.

    Parameters
    ----------
    state : SpringState
        The current spring state.
    params : SpringParams
        The parameters of the spring model.
    sign : jnp.ndarray
        The sign of the edges.
    edge_index : jnp.ndarray
        The edge index of the graph.
    """

    position_i = state.position[edge_index[0]]
    position_j = state.position[edge_index[1]]

    edge_forces = compute_force(params, position_i, position_j, sign)
    node_forces = jnp.zeros_like(state.position)
    node_forces = node_forces.at[edge_index[0]].add(edge_forces)

    velocity = state.velocity + 0.5 * params.time_step * node_forces
    position = state.position + params.time_step * velocity

    edge_forces = compute_force(params, position_i, position_j, sign)
    node_forces = jnp.zeros_like(state.position)
    node_forces = node_forces.at[edge_index[0]].add(edge_forces)

    velocity = velocity + 0.5 * params.time_step * node_forces

    velocity = velocity * (1.0 - params.damping)

    energy = jnp.sum(jnp.square(velocity), axis=1)

    return SpringState(position, velocity, energy)


@partial(jax.jit, static_argnames=["iterations"])
def simulate(
    iterations : int,
    spring_state : SpringState,
    spring_params : SpringParams,
    signs : jnp.ndarray,
    edge_index : jnp.ndarray) -> SpringState:

    spring_state = jax.lax.fori_loop(
        0, 
        iterations, 
        lambda i, 
        state: update(spring_state, spring_params, signs, edge_index), 
        spring_state)

    embeddings = spring_state.position
    position_i = embeddings.at[edge_index[0]].get()
    position_j = embeddings.at[edge_index[1]].get()

    spring_vec = position_i - position_j
    spring_vec_norm = jnp.linalg.norm(spring_vec, axis=1) - NEUTRAL_DISTANCE

    sigmoid = lambda x: 1.0 / (1.0 + jnp.exp(-x))

    return jnp.sum(sigmoid(spring_vec_norm) - signs), spring_state