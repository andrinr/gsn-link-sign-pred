from typing import NamedTuple
import jax.numpy as jnp
import jax

class HeuristicForceParams(NamedTuple):
    friend_distance: float
    friend_stiffness: float
    neutral_distance: float
    neutral_stiffness: float
    enemy_distance: float
    enemy_stiffness: float
    center_attraction: float

class SpringState(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray
    acceleration: jnp.ndarray
    
def init_spring_state(
    rng : jax.random.PRNGKey, 
    n : int, m : int,
    range : float,
    embedding_dim : int) -> SpringState:
    position = jax.random.uniform(rng, (n, embedding_dim), minval=-range, maxval=range)
    velocity = jnp.zeros((n, embedding_dim))
    acceleration = jnp.zeros((n, embedding_dim))
    return SpringState(position, velocity, acceleration)

class SimulationState(NamedTuple):
    iteration : int
    time : float

class SimulationParams(NamedTuple):
    iterations : int
    dt : float
    damping : float
    message_passing_iterations : int