from typing import NamedTuple
import jax.numpy as jnp
import jax

class SpringParams(NamedTuple):
    friend_distance: float
    friend_stiffness: float
    neutral_distance: float
    neutral_stiffness: float
    enemy_distance: float
    enemy_stiffness: float
    distance_threshold: float

class SpringState(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray
    auxillary: jnp.ndarray
    force_decision: jnp.ndarray
    
def init_spring_state(
    rng : jax.random.PRNGKey, 
    n : int, m : int,
    range : float,
    embedding_dim : int,
    auxillary_dim : int) -> SpringState:
    position = jax.random.uniform(rng, (n, embedding_dim), maxval=range, minval=-range)
    velocity = jnp.zeros((n, embedding_dim))
    auxillary = jnp.zeros((n, auxillary_dim))
    force_decision = jnp.zeros((m, 3))
    return SpringState(position, velocity, auxillary, force_decision)

class SimulationState(NamedTuple):
    iteration : int
    time : float

class SimulationParams(NamedTuple):
    iterations : int
    dt : float
    damping : float
    message_passing_iterations : int