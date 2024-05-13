from typing import NamedTuple
import jax.numpy as jnp
import jax
from jax import random

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

def init_neural_force_params() -> NeuralForceParams:
    initializer = jax.nn.initializers.normal()
    c = 0.01
    friend=MLP(
        w0=initializer(random.PRNGKey(0), (7, 4)) * c,
        b0=jnp.zeros(4),
        w1=initializer(random.PRNGKey(1), (4,1)) * c,
        b1=jnp.zeros(1))
    
    neutral=MLP(
        w0=initializer(random.PRNGKey(2), (7, 4)) * c,
        b0=jnp.zeros(4),
        w1=initializer(random.PRNGKey(3), (4,1)) * c,
        b1=jnp.zeros(1))
    
    enemy=MLP(
        w0=initializer(random.PRNGKey(4), (7, 4)) * c,
        b0=jnp.zeros(4),
        w1=initializer(random.PRNGKey(5), (4,1)) * c,
        b1=jnp.zeros(1))
    
    return NeuralForceParams(friend, neutral, enemy)

class SpringForceParams(NamedTuple):
    friend_distance: float
    friend_stiffness: float
    neutral_distance: float
    neutral_stiffness: float
    enemy_distance: float
    enemy_stiffness: float
    degree_multiplier: float

def init_spring_force_params() -> SpringForceParams:
    return SpringForceParams(
        friend_distance=1.0,
        friend_stiffness=1.0,
        neutral_distance=1.0,
        neutral_stiffness=0.1,
        enemy_distance=5.5,
        enemy_stiffness=2.0,
        degree_multiplier=3.0)

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