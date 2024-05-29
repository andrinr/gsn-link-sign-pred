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
    init_orth = jax.nn.initializers.orthogonal()
    init_norm = jax.nn.initializers.normal()
    c = 0.001
    n_in = 7
    n_hidden = 7

    mlps = []

    for i in range(3):
        mlps.append(MLP(
            w0=init_orth(random.PRNGKey(i), (n_in, n_hidden)),
            b0=jnp.zeros(n_hidden),
            w1=init_norm(random.PRNGKey(i+1), (n_hidden + n_in, 1)) * c,
            b1=jnp.zeros(1)))
    
    return NeuralForceParams(
        friend=mlps[0],
        neutral=mlps[1],
        enemy=mlps[2])

def evaluate_mlp(mlp : MLP, x : jnp.ndarray) -> jnp.ndarray:
    skip_state = x[:, :]
    x = jnp.dot(x, mlp.w0) + mlp.b0
    x = jax.nn.tanh(x)
    x = jnp.concatenate([x, skip_state], axis=1)
    x = jnp.dot(x, mlp.w1) + mlp.b1
    return x

class SpringForceParams(NamedTuple):
    friend_distance: float
    friend_stiffness: float
    neutral_distance: float
    neutral_stiffness: float
    enemy_distance: float
    enemy_stiffness: float
    degree_multiplier: float

class SpringState(NamedTuple):
    position : jnp.ndarray
    velocity : jnp.ndarray

class SimulationState(NamedTuple):
    iteration : int
    time : float

class SimulationParams(NamedTuple):
    iterations : int
    dt : float
    damping : float

def init_spring_force_params() -> SpringForceParams:
    return SpringForceParams(
        friend_distance=1.0,
        friend_stiffness=1.0,
        neutral_distance=1.0,
        neutral_stiffness=0.1,
        enemy_distance=5.5,
        enemy_stiffness=2.0,
        degree_multiplier=3.0)

def init_spring_state(
    rng : jax.random.PRNGKey, 
    n : int, m : int,
    range : float,
    embedding_dim : int) -> SpringState:
    position = jax.random.uniform(rng, (n, embedding_dim), minval=-range, maxval=range)
    velocity = jnp.zeros_like(position)
    return SpringState(position, velocity)