import jax
import jax.numpy as jnp
from typing import NamedTuple

class NeuralForceParams(NamedTuple):
    W0 : jnp.ndarray
    b0 : jnp.ndarray
    W1 : jnp.ndarray
    b1 : jnp.ndarray
    W2 : jnp.ndarray
    b2 : jnp.ndarray
    W3 : jnp.ndarray
    b3 : jnp.ndarray
    W4 : jnp.ndarray
    b4 : jnp.ndarray

def init_neural_force_params(
    key : jax.random.PRNGKey,
    factor : float) -> NeuralForceParams:

    keys = jax.random.split(key, num=5)
    
    sizes = [4, 32, 32, 32, 16, 1]

    params = {}

    for i in range(len(sizes) - 1):
        params[f'W{i}'] = jax.random.normal(keys[i], (sizes[i], sizes[i+1])) * factor
        params[f'b{i}'] = jnp.zeros((sizes[i+1],))

    return NeuralForceParams(**params)

@jax.jit
def mlp_forces(
    x : jnp.ndarray,
    params : dict[jnp.ndarray]) -> jnp.ndarray:

    x = jnp.dot(x, params.W0) + params.b0
    x = jax.nn.relu(x)

    x = jnp.dot(x, params.W1) + params.b1
    x = jax.nn.relu(x)

    x = jnp.dot(x, params.W2) + params.b2
    x = jax.nn.relu(x)

    x = jnp.dot(x, params.W3) + params.b3
    x = jax.nn.relu(x)

    x = jnp.dot(x, params.W4) + params.b4
    x = jax.nn.tanh(x)

    return x
    