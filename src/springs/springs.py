import jax.numpy as jnp
import jax
from typing import NamedTuple
from functools import partial

NEUTRAL_DISTANCE = 10.0
NEUTRAL_STIFFNESS = 10.0

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

def init_spring_state(rng : jax.random.PRNGKey, n : int, embedding_dim : int) -> SpringState:
    position = jax.random.uniform(rng, (n, embedding_dim), maxval=1.0, minval=-1.0)
    velocity = jnp.zeros((n, embedding_dim))
    energy = jnp.zeros(n)

    return SpringState(position, velocity, energy)

# @partial(jax.jit)
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

# @partial(jax.jit)
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

    state = state._replace(
        position=position,
        velocity=velocity,
        energy=energy)
    
    return state

@partial(jax.jit, static_argnames=["iterations"])
def simulate(
    iterations : int,
    spring_state : SpringState,
    spring_params : SpringParams,
    signs : jnp.ndarray,
    edge_index : jnp.ndarray) -> SpringState:
    """
    Simulate the spring model for a number of iterations.

    Parameters
    ----------
    iterations : int
        The number of iterations.
    spring_state : SpringState
        The current spring state.
    spring_params : SpringParams
        The parameters of the spring model.
    signs : jnp.ndarray
        The sign of the edges.
    edge_index : jnp.ndarray
        The edge index of the graph.
    """

    spring_state = jax.lax.fori_loop(
        0, 
        iterations, 
        # capture the spring_params and signs in the closure
        lambda i, spring_state: update(spring_state, spring_params, signs, edge_index), 
        spring_state)
    
    return spring_state

@partial(jax.jit, static_argnames=["iterations"])
def simulate_and_loss(
    iterations : int,
    spring_state : SpringState,
    spring_params : SpringParams,
    signs : jnp.ndarray,
    training_mask : jnp.ndarray,
    validation_mask : jnp.ndarray,
    edge_index : jnp.ndarray) -> SpringState:
    """
    Simulate the spring model for a number of iterations and compute the loss.

    Parameters
    ----------
    iterations : int
        The number of iterations.
    spring_state : SpringState
        The current spring state.
    spring_params : SpringParams
        The parameters of the spring model.
    signs : jnp.ndarray
        The sign of the edges.
    training_mask : jnp.ndarray 
        The training mask.
    validation_mask : jnp.ndarray
        The validation mask.
    edge_index : jnp.ndarray
        The edge index of the graph.
    """

    training_signs = jnp.where(training_mask, signs, 0)

    spring_state = simulate(
        iterations,
        spring_state,
        spring_params,
        training_signs,
        edge_index)

    embeddings = spring_state.position
    position_i = embeddings.at[edge_index[0]].get()
    position_j = embeddings.at[edge_index[1]].get()

    spring_vec = position_i - position_j
    spring_vec_norm = jnp.linalg.norm(spring_vec, axis=1)
    
    predicted_sign = spring_vec_norm - NEUTRAL_DISTANCE
    logistic = lambda x: 1 / (1 + jnp.exp(-x))
    predicted_sign = logistic(predicted_sign)

    signs = jnp.where(signs == 1, 1, 0)

    print(predicted_sign.shape)

    # categorical loss
    loss = jnp.square(predicted_sign - signs)
    loss = jnp.where(validation_mask, loss, 0)
    loss = jnp.sum(loss)

    return loss, spring_state
