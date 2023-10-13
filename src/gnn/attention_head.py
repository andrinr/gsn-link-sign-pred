import numpy as np
import jax.numpy as jnp
import flax.linen as nn

class AttentionHead(nn.Module):
  def __init__(
    self,
    n_dimensions : int
  ): 
    super().__init__()
    self.n_dimensions = n_dimensions
    self.input_shape = 3 * n_dimensions + 1

    self.Q = nn.Dense(features=self.input_shape)
    self.K = nn.Dense(features=self.input_shape)
    self.V = nn.Dense(features=self.input_shape)

  @nn.compact
  def __call__(
        self, 
        node_position_i : jnp.ndarray,
        node_position_j : jnp.ndarray,
        node_information_i : jnp.ndarray,
        node_information_j : jnp.ndarray,
        sign : jnp.ndarray) -> jnp.ndarray:
    
    other_node = jnp.concatenate([node_position_j, node_information_j, sign], axis=-1)
    self_node = jnp.concatenate([node_position_i, node_information_i, sign], axis=-1)

    q = self.Q(self_node)
    k = self.K(other_node)
    v = self.V(other_node)

    score_softmax = jnp.dot(q, k.T) / jnp.sqrt(self.n_dimensions)

    return score_softmax * v