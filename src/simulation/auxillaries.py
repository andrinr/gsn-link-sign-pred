import jax.numpy as jnp
import jax
from functools import partial
import simulation as sim
import neural as nn

@partial(jax.jit)
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

    sign = jnp.expand_dims(sign, axis=1)

    auxillaries_i = nn.attention(
        jnp.concatenate([auxillaries_i, sign], axis=-1),
        jnp.concatenate([auxillaries_j, sign], axis=-1),
        auxillaries_nn_params)
    
    auxillaries = jnp.zeros_like(spring_state.auxillary)
    auxillaries = auxillaries.at[edge_index[0]].add(auxillaries_i)

    return sim.SpringState(
        position=spring_state.position,
        velocity=spring_state.velocity,
        auxillary=auxillaries)