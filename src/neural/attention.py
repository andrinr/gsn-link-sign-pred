import jax
import jax.numpy as jnp
from jax.experimental import checkify

def init_attention_params(
  key : jax.random.PRNGKey,
  input_dimension : int,
  output_dimension : int,
  factor : float = 0.1) -> dict[jnp.ndarray]:

  key0, key1, key2 = jax.random.split(key, num=3)

  return {
    'Q': jax.random.normal(key=key0, shape=(input_dimension, output_dimension)) * factor,
    'K': jax.random.normal(key=key1, shape=(input_dimension, output_dimension)) * factor,
    'V': jax.random.normal(key=key2, shape=(input_dimension, output_dimension)) * factor
  }

@jax.jit
def attention(
    x_i : jnp.ndarray,
    x_j : jnp.ndarray,
    params : dict[jnp.ndarray]) -> jnp.ndarray:

  q = jnp.dot(x_i, params['Q'])
  k = jnp.dot(x_j, params['K'])
  v = jnp.dot(x_j, params['V'])
  
  score_softmax = jnp.einsum('ij,ij->i', q, k) / jnp.sqrt(params['Q'].shape[1])
  v = jnp.einsum('i,ij->ij', score_softmax, v)
  
  return v
