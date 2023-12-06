from typing import NamedTuple
import jax.numpy as jnp
import jax

class HeuristicForceParams(NamedTuple):
    friend_intercept : jnp.ndarray
    enemy_intercept : jnp.ndarray
    neutral_intercept : jnp.ndarray
    friend_slope : jnp.ndarray
    enemy_slope : jnp.ndarray
    neutral_slope : jnp.ndarray
    friend_segment : jnp.ndarray
    enemy_segment : jnp.ndarray
    neutral_segment : jnp.ndarray

class SpringState(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray
    
def init_spring_state(
    rng : jax.random.PRNGKey, 
    n : int, m : int,
    range : float,
    embedding_dim : int) -> SpringState:
    position = jax.random.uniform(rng, (n, embedding_dim), minval=-range, maxval=range)
    velocity = jnp.zeros((n, embedding_dim))
    return SpringState(position, velocity)

class SimulationState(NamedTuple):
    iteration : int
    time : float

class SimulationParams(NamedTuple):
    iterations : int
    dt : float
    damping : float
    centering : float