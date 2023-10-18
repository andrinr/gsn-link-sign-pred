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
    Update the spring state using the leapfrog method. 
    This is essentially a simple message passing network implementation. 
    """

    auxillaries_i = spring_state.auxillaries[edge_index[0]]
    auxillaries_j = spring_state.auxillaries[edge_index[1]]

    sign = jnp.expand_dims(sign, axis=1)

    auxillaries_i = nn.attention(
        jnp.concatenate([auxillaries_i, sign], axis=-1),
        jnp.concatenate([auxillaries_j, sign], axis=-1),
        auxillaries_nn_params)
    
    auxillaries = jnp.zeros_like(spring_state.auxillaries)
    auxillaries = auxillaries.at[edge_index[0]].add(auxillaries_i)
    
    spring_state = spring_state._replace(
        auxillaries=auxillaries)
    
    return spring_state