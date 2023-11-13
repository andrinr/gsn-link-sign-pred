import jax
import jax.numpy as jnp

def init_force_params(
    key : jax.random.PRNGKey,
    factor : float,    
    auxillary_dim : int) -> dict[jnp.ndarray]:

    keys = jax.random.split(key, num=2)

    params = {}

    # layer 0
    params[f'W{0}'] = jax.random.normal(
        key=keys[0], shape=(auxillary_dim * 2 + 3, auxillary_dim * 2 + 3), dtype=jnp.float32) * factor
    params[f'b{0}'] = jnp.zeros(auxillary_dim * 2 + 3, dtype=jnp.float32)

    # layer 1
    params[f'W{1}'] = jax.random.normal(
        key=keys[1], shape=(auxillary_dim * 2 + 3, auxillary_dim), dtype=jnp.float32) * factor
    params[f'b{1}'] = jnp.zeros(auxillary_dim, dtype=jnp.float32)

    # layer 2
    params[f'W{2}'] = jax.random.normal(
        key=keys[1], shape=(auxillary_dim, 3), dtype=jnp.float32) * factor
    params[f'b{2}'] = jnp.zeros(3, dtype=jnp.float32)

    return params

@jax.jit
def mlp_forces(
    x : jnp.ndarray,
    params : dict[jnp.ndarray]) -> jnp.ndarray:

    x = jnp.dot(x, params[f'W{0}']) + params[f'b{0}']
    x = jax.nn.relu(x)

    x = jnp.dot(x, params[f'W{1}']) + params[f'b{1}']
    x = jax.nn.relu(x)

    x = jnp.dot(x, params[f'W{2}']) + params[f'b{2}']
    x = jax.nn.softmax(x)

    return x
    