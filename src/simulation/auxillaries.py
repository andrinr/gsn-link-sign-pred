import jax.numpy as jnp
import jax
from functools import partial
import simulation as sim
import neural as nn
from graph import SignedGraph

EPSILON = 1e-6

# @partial(jax.jit)
def update_auxillary_state(
    spring_state : sim.SpringState, 
    auxillaries_nn_params : dict[jnp.ndarray],
    graph : SignedGraph) -> sim.SpringState:
    """
    Update the auxillary state using the message passing method.
    """

    auxillaries_i = spring_state.auxillary[graph.edge_index[0]]
    auxillaries_j = spring_state.auxillary[graph.edge_index[1]]
    degs_i = jnp.expand_dims(graph.node_degrees[graph.edge_index[0]], axis=-1)
    degs_j = jnp.expand_dims(graph.node_degrees[graph.edge_index[1]], axis=-1)

    max_deg = jnp.maximum(degs_i, degs_j)

    degs_i = degs_i / max_deg
    degs_j = degs_j / max_deg

    sign_one_hot = jax.nn.one_hot(graph.sign, 3)
 
    auxillaries_i = nn.gnn_psi(
        jnp.concatenate([auxillaries_i, auxillaries_j, sign_one_hot, degs_i, degs_j], axis=-1),
        auxillaries_nn_params)

    # mean aggregation
    auxillaries = jnp.zeros_like(spring_state.auxillary)
    auxillaries = auxillaries.at[graph.edge_index[0]].add(auxillaries_i)
    # avoid division by zero
    # auxillaries = auxillaries / jnp.expand_dims(graph.node_degrees + EPSILON , axis=-1)
   
    auxillaries = nn.gnn_phi(
        jnp.concatenate([auxillaries, spring_state.auxillary], axis=-1),
        auxillaries_nn_params)
    
    return sim.SpringState(
        position=spring_state.position,
        velocity=spring_state.velocity,
        auxillary=auxillaries,
        force_decision=spring_state.force_decision)