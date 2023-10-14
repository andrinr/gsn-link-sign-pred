import jax.numpy as jnp
import jax
from typing import NamedTuple, Callable

class GraphParams(NamedTuple):
    node_embedding_dimensions : int
    edge_embedding_dimensions : int
    num_edges : int
    num_nodes : int

def propagate(
    edge_index : jnp.ndarray,
    node_data : jnp.ndarray,
    updated_data : jnp.ndarray,
    update_function : Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    scatter_method : str
    ) -> jnp.ndarray:

    node_data_i = node_data[edge_index[0]]
    node_data_j = node_data[edge_index[1]]

    updated_data = update_function(node_data_i, node_data_j, updated_data)

    if scatter_method == "sum":
        node_data = node_data.at[edge_index[0]].add(updated_data)
    elif scatter_method == "mean":
        node_data = node_data.at[edge_index[0]].add(updated_data)
    elif scatter_method == "max":
        node_data = node_data.at[edge_index[0]].max(updated_data)
    else:
        raise ValueError(f"Unknown scatter method {scatter_method}")