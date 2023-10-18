import jax.numpy as jnp
import jax
from typing import NamedTuple
from functools import partial
from gnn import AttentionHead
import optax

@partial(jax.jit, static_argnames=["key", "message_passing_iterations", "embedding_dim", "learning_rate"])
def generate_auxillaries(
        key : jax.random.PRNGKey,
        message_passing_iterations : int,
        embedding_dim : int,
        learning_rate : float,
        edge_index : jnp.ndarray,
        signs : jnp.ndarray,) -> jnp.ndarray:
    
    num_edges = edge_index.shape[1]
    auxillaries = jax.random.uniform(key, (num_edges, embedding_dim), maxval=1.0, minval=-1.0)

    attention_head = AttentionHead(embedding_dimensions=embedding_dim)
    params = attention_head.init(key)

    tx = optax.adam(learning_rate=learning_rate)


    