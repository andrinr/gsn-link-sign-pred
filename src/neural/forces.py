import jax
import jax.numpy as jnp

def init_force_params(
    key : jax.random.PRNGKey,
    factor : float) -> dict[jnp.ndarray]:

    keys = jax.random.split(key, num=5)

    params = {}

    input_dim = 3 + 1 + 1
    # layer 0
    params[f'W{0}'] = jax.random.normal(
        key=keys[0], shape=(input_dim, input_dim * 4), dtype=jnp.float32) * factor
    params[f'b{0}'] = jnp.zeros(input_dim * 4, dtype=jnp.float32)

    # layer 1
    params[f'W{1}'] = jax.random.normal(
        key=keys[1], shape=(input_dim * 4, input_dim * 4), dtype=jnp.float32) * factor
    params[f'b{1}'] = jnp.zeros(input_dim * 4, dtype=jnp.float32)

    # layer 2
    params[f'W{2}'] = jax.random.normal(
        key=keys[2], shape=(input_dim * 4, input_dim * 2), dtype=jnp.float32) * factor
    params[f'b{2}'] = jnp.zeros(input_dim * 2, dtype=jnp.float32)

    # layer 3
    params[f'W{3}'] = jax.random.normal(
        key=keys[3], shape=(input_dim * 2, input_dim), dtype=jnp.float32) * factor
    params[f'b{3}'] = jnp.zeros(input_dim, dtype=jnp.float32)

    # layer 4
    params[f'W{4}'] = jax.random.normal(
        key=keys[4], shape=(input_dim, 1), dtype=jnp.float32) * factor
    params[f'b{4}'] = jnp.zeros(1, dtype=jnp.float32)

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
    x = jax.nn.relu(x)

    x = jnp.dot(x, params[f'W{3}']) + params[f'b{3}']
    x = jax.nn.relu(x)

    x = jnp.dot(x, params[f'W{4}']) + params[f'b{4}']

    return x
    