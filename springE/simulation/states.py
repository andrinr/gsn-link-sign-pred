from typing import NamedTuple
import jax.numpy as jnp
import jax

class Spring(NamedTuple):
    attraction_stiffness: float
    repulsion_stiffness: float
    rest_length: float
    degree_i_multiplier: float
    degree_j_multiplier: float  

class MLP(NamedTuple):
    w0 : jnp.ndarray
    w1 : jnp.ndarray
    b0 : jnp.ndarray
    b1 : jnp.ndarray

class NeuralForceParams(NamedTuple):
    friend : MLP
    neutral : MLP
    enemy : MLP

class SpringForceParams(NamedTuple):
    friend_distance: float
    friend_stiffness: float
    neutral_distance: float
    neutral_stiffness: float
    enemy_distance: float
    enemy_stiffness: float
    degree_multiplier: float

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