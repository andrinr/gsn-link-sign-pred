import numpy as np
import jax.numpy as jnp
import flax.linen as nn

class AttentionHead(nn.Module):
  def setup(
    self,
    input_dimension : int,
    embedding_dimensions : int
  ): 
    self.input_dimension = input_dimension
    self.embedding_dimensions = embedding_dimensions

    self.Q = nn.Dense(features=embedding_dimensions)
    self.K = nn.Dense(features=embedding_dimensions)
    self.V = nn.Dense(features=embedding_dimensions)

  def __call__(
        self, 
        node_data_i : jnp.ndarray,
        node_data_j : jnp.ndarray,
        sign : jnp.ndarray) -> jnp.ndarray:
    
    other_node = jnp.concatenate([node_data_j, sign], axis=-1)
    self_node = jnp.concatenate([node_data_i, sign], axis=-1)

    q = self.Q(self_node)
    k = self.K(other_node)
    v = self.V(other_node)

    score_softmax = jnp.dot(q, k.T) / jnp.sqrt(self.embedding_dimensions)

    return score_softmax * v