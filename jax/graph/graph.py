import jax.numpy as jnp
from typing import Optional

class Graph:

    def __init__(
        self, 
        edge_index: jnp.ndarray,
        edge_attr: Optional[jnp.ndarray] = None,
        node_attr: Optional[jnp.ndarray] = None) -> None:

        """
        Graph data structure for JAX.

        Args:
            edge_index (jnp.ndarray): Edge index tensor of shape (2, num_edges).
            edge_attr (jnp.ndarray, optional): Edge attribute tensor of shape (num_edges, num_edge_features). Defaults to None.
            node_attr (jnp.ndarray, optional): Node attribute tensor of shape (num_nodes, num_node_features). Defaults to None.

        """

        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.node_attr = node_attr

        self.num_nodes = jnp.max(self.edge_index) + 1
        self.num_edges = self.edge_index.shape[0]

        self.check()

    def check(self) -> None:

        if self.edge_attr is not None:
            assert self.edge_attr.shape[0] == self.num_edges

        if self.node_attr is not None:
            assert self.node_attr.shape[0] == self.num_nodes
