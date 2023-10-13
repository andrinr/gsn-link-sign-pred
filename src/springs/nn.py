from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

class Attention(nn.Module):
  
  def __init__(
    self,
    n_dimensions : int
  ): 
    super().__init__()
    self.n_dimensions = n_dimensions
    self.input_shape = 3 * n_dimensions + 1

  @nn.compact
  def __call__(
        self, 
        node_embedding_i : jnp.ndarray,
        node_embedding_j : jnp.ndarray,
        sign : jnp.ndarray,
        edge_embedding : jnp.ndarray) -> jnp.ndarray:
    
    x = jnp.concatenate([node_embedding_i, node_embedding_j, sign, edge_embedding], axis=-1)

    x = nn.Dense(features=self.input_shape)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.input_shape)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.n_dimensions)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.n_dimensions)(x)
    x = nn.relu(x)

    return x