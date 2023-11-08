import jax.numpy as jnp
import jax
from functools import partial
import simulation as sim
import neural as nn

# @partial(jax.jit)
def update_auxillary_state(
    spring_state : sim.SpringState, 
    auxillaries_nn_params : dict[jnp.ndarray],
    edge_index : jnp.ndarray,
    sign : jnp.ndarray) -> sim.SpringState:
    """
    Update the auxillary state using the message passing method.
    """

    auxillaries_i = spring_state.auxillary[edge_index[0]]
    auxillaries_j = spring_state.auxillary[edge_index[1]]

    sign_one_hot = jax.nn.one_hot(sign, 3)

    auxillaries_i = nn.gnn_psi(
        jnp.concatenate([auxillaries_i, auxillaries_j, sign_one_hot], axis=-1),
        auxillaries_nn_params)

    # mean aggregation
    auxillaries = jnp.zeros_like(spring_state.auxillary)
    auxillaries = auxillaries.at[edge_index[0]].add(auxillaries_i)
    auxillaries = auxillaries / jnp.expand_dims(jnp.bincount(edge_index[0]), axis=-1)
   
    auxillaries = nn.gnn_phi(
        jnp.concatenate([auxillaries, spring_state.auxillary], axis=-1),
        auxillaries_nn_params)
    
    return sim.SpringState(
        position=spring_state.position,
        velocity=spring_state.velocity,
        auxillary=auxillaries,
        force_decision=spring_state.force_decision)