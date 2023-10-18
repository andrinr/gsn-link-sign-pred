import jax
import jax.numpy as jnp

def init_attention_params(
  key : jax.random.PRNGKey,
  input_dimension : int,
  output_dimension : int) -> dict[jnp.ndarray]:

  key0, key1, key2 = jax.random.split(key, num=3)

  return {
    'Q': jax.random.normal(shape=(input_dimension, output_dimension), key=key0),
    'K': jax.random.normal(shape=(input_dimension, output_dimension), key=key1),
    'V':jax.random.normal(shape=(input_dimension, output_dimension), key=key2),
  }

@jax.jit
def attention(
    x_i : jnp.ndarray,
    x_j : jnp.ndarray,
    params : dict[jnp.ndarray]) -> jnp.ndarray:
  
  q = jnp.dot(x_i, params['Q'])
  k = jnp.dot(x_j, params['K'])
  v = jnp.dot(x_j, params['V'])

  score_softmax = jnp.dot(q, k.T) / jnp.sqrt(params['Q'].shape[1])

  return score_softmax * v