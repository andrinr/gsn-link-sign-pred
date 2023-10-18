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
def compute_force(
    position_i : jnp.ndarray,
    position_j : jnp.ndarray,
    auxillaries_i : jnp.ndarray,
    auxillaries_j : jnp.ndarray,
    sign : jnp.ndarray,
    params : dict[jnp.ndarray]) -> jnp.ndarray:
  
  vector = position_j - position_i
  
  other_node = jnp.concatenate([position_i, auxillaries_i, sign], axis=-1)
  self_node = jnp.concatenate([position_j, auxillaries_j], axis=-1)

  q = jnp.dot(self_node, params['Q'])
  k = jnp.dot(other_node, params['K'])
  v = jnp.dot(other_node, params['V'])

  score_softmax = jnp.dot(q, k.T) / jnp.sqrt(params['Q'].shape[1])

  return score_softmax * v