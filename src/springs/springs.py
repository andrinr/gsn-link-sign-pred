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

class SpringState(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray

def init_spring_state(rng : jax.random.PRNGKey, n : int, embedding_dim : int) -> SpringState:
    position = jax.random.uniform(rng, (n, embedding_dim), maxval=1.0, minval=-1.0)
    velocity = jnp.zeros((n, embedding_dim))

    return SpringState(position, velocity)

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
    dt : float,
    damping : float,
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

    velocity = state.velocity + 0.5 * dt * node_forces
    position = state.position + dt * velocity

    edge_forces = compute_force(params, position_i, position_j, sign)
    node_forces = jnp.zeros_like(state.position)
    node_forces = node_forces.at[edge_index[0]].add(edge_forces)

    velocity = velocity + 0.5 * dt * node_forces

    velocity = velocity * (1.0 - damping * dt)

    state = state._replace(
        position=position,
        velocity=velocity)
    
    return state

@partial(jax.jit, static_argnames=["iterations"])
def simulate(
    iterations : int,
    spring_state : SpringState,
    spring_params : SpringParams,
    dt : float,
    damping : float,
    signs : jnp.ndarray,
    edge_index : jnp.ndarray) -> SpringState:

    spring_state = jax.lax.fori_loop(
        0, 
        iterations, 
        # capture the spring_params and signs in the closure
        lambda i, spring_state: update(spring_state, spring_params, dt, damping, signs, edge_index), 
        spring_state)
    
    return spring_state

@jax.jit
def f_beta(truth : jnp.array, prediction : jnp.array, beta : float) -> float:
    tp = jnp.sum(prediction * truth)
    fp = jnp.sum(prediction * (1 - truth))
    fn = jnp.sum((1 - prediction) * truth)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

@jax.jit
def f1_macro(truth : jnp.array, prediction : jnp.array) -> float:
    tp = jnp.sum(prediction * truth, axis=0)
    fp = jnp.sum(prediction * (1 - truth), axis=0)
    fn = jnp.sum((1 - prediction) * truth, axis=0)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return jnp.mean(2 * precision * recall / (precision + recall))

@partial(jax.jit, static_argnames=["iterations"])
def simulate_and_loss(
    iterations : int,
    spring_state : SpringState,
    spring_params : SpringParams,
    dt : float,
    damping : float,
    signs : jnp.ndarray,
    training_mask : jnp.ndarray,
    validation_mask : jnp.ndarray,
    edge_index : jnp.ndarray) -> SpringState:

    training_signs = signs.copy()
    training_signs = jnp.where(training_mask, training_signs, 0)

    spring_state = simulate(
        iterations,
        spring_state,
        spring_params,
        dt,
        damping,
        training_signs,
        edge_index)

    position_i = spring_state.position[edge_index[0]]
    position_j = spring_state.position[edge_index[1]]

    spring_vec_norm = jnp.linalg.norm(position_i - position_j, axis=1)
    
    predicted_sign = spring_vec_norm - NEUTRAL_DISTANCE
    logistic = lambda x: 1 / (1 + jnp.exp(-x))
    predicted_sign = logistic(predicted_sign)

    signs = jnp.where(signs == 1, 1, 0)

    fraction_negatives = jnp.mean(predicted_sign)
    fraction_positives = 1 - fraction_negatives

    score = 1/fraction_positives * signs * predicted_sign + 1/fraction_negatives * (1 - signs) * (1 - predicted_sign)

    loss = -jnp.mean(score)
    
    return loss, spring_state
