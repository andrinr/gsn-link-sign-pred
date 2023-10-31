import jax
import jax.numpy as jnp

def init_mlp_params(
    key : jax.random.PRNGKey,
    layer_dimensions : list[int]) -> dict[jnp.ndarray]:

    keys = jax.random.split(key, num=len(layer_dimensions))

    params = {}

    for i, (key, (input_dimension, output_dimension)) in enumerate(zip(keys, zip(layer_dimensions[:-1], layer_dimensions[1:]))):
        params[f'W{i}'] = jax.random.normal(key=key, shape=(input_dimension, output_dimension))
        params[f'b{i}'] = jnp.zeros(output_dimension)

    return params

@jax.jit
def mlp(
    x : jnp.ndarray,
    params : dict[jnp.ndarray]) -> jnp.ndarray:

    for i in range(len(params) // 2):
        x = jnp.dot(x, params[f'W{i}']) + params[f'b{i}']
        if i < len(params) // 2 - 1:
            x = jax.nn.relu(x)

    return x