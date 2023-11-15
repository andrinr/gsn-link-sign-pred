import jax
import jax.numpy as jnp

def init_gnn_params(
    key : jax.random.PRNGKey,
    factor : float,
    auxilliary_dimension : int) -> dict[jnp.ndarray]:

    keys = jax.random.split(key, num=4)

    params = {}
    
    # psi
    psi_input_dimension = auxilliary_dimension * 2 + 3 + 2
    params[f'W{0}_psi'] = jax.random.normal(
        key=keys[0], shape=(psi_input_dimension, psi_input_dimension), dtype=jnp.float32) * factor
    params[f'b{0}_psi'] = jnp.zeros(psi_input_dimension, dtype=jnp.float32)
    params[f'W{1}_psi'] = jax.random.normal(
        key=keys[1], shape=(psi_input_dimension, auxilliary_dimension), dtype=jnp.float32) * factor
    params[f'b{1}_psi'] = jnp.zeros(auxilliary_dimension, dtype=jnp.float32)

    # phi
    phi_input_dimension = auxilliary_dimension * 2
    params[f'W{0}_phi'] = jax.random.normal(
        key=keys[2], shape=(phi_input_dimension, phi_input_dimension), dtype=jnp.float32) * factor
    params[f'b{0}_phi'] = jnp.zeros(phi_input_dimension, dtype=jnp.float32)
    params[f'W{1}_phi'] = jax.random.normal(
        key=keys[3], shape=(phi_input_dimension, auxilliary_dimension), dtype=jnp.float32) * factor
    params[f'b{1}_phi'] = jnp.zeros(auxilliary_dimension, dtype=jnp.float32)

    return params

@jax.jit
def gnn_psi(
    x : jnp.ndarray,
    params : dict[jnp.ndarray]) -> jnp.ndarray:

    x = jnp.dot(x, params[f'W{0}_psi']) + params[f'b{0}_psi']
    x = jax.nn.relu(x)

    x = jnp.dot(x, params[f'W{1}_psi']) + params[f'b{1}_psi']
    x = jax.nn.relu(x)

    return x

@jax.jit
def gnn_phi(
    x : jnp.ndarray,
    params : dict[jnp.ndarray]) -> jnp.ndarray:

    x = jnp.dot(x, params[f'W{0}_phi']) + params[f'b{0}_phi']
    x = jax.nn.relu(x)

    x = jnp.dot(x, params[f'W{1}_phi']) + params[f'b{1}_phi']
    x = jax.nn.relu(x)

    return x