import jax.numpy as jnp
import jax
from typing import NamedTuple
from functools import partial

class SpringParams(NamedTuple):
    friend_distance: float
    friend_stiffness: float
    neutral_distance: float
    neutral_stiffness: float
    enemy_distance: float
    enemy_stiffness: float
    damping: float
    time_step: float

class SpringState(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray
    energy: jnp.ndarray
    loss : float = 0.0

def init_spring_state(rng : jax.random.PRNGKey, n : int, embedding_dim : int) -> SpringState:
    position = jax.random.uniform(rng, (n, embedding_dim), maxval=1.0, minval=-1.0)
    velocity = jnp.zeros((n, embedding_dim))
    energy = jnp.zeros(n)

    return SpringState(position, velocity, energy)

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def compute_force(
    params : SpringParams, 
    position_i : jnp.ndarray,
    position_j : jnp.ndarray,
    sign : jnp.ndarray) -> jnp.ndarray:
    
    spring_vector = position_j - position_i
    l = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (l + 0.001)

    attraction = jnp.maximum(l - params.friend_distance, 0) * params.friend_stiffness * spring_vector_norm
    neutral = (l - params.neutral_distance) * params.neutral_stiffness * spring_vector_norm
    retraction = -jnp.maximum(params.enemy_distance - l, 0) * params.enemy_stiffness * spring_vector_norm

    sign = jnp.expand_dims(sign, axis=1)
    force = jnp.where(sign == 1, attraction, retraction)
    force = jnp.where(sign == 0, neutral, force)
    
    return force

@partial(jax.jit, static_argnums=(1, 2, 3))
def update(
    state : SpringState, 
    params : SpringParams, 
    sign : jnp.ndarray, 
    edge_index : jnp.ndarray) -> SpringState:
    """
    Update the spring state using the leapfrog method.

    Parameters
    ----------
    state : SpringState
        The current spring state.
    params : SpringParams
        The parameters of the spring model.
    sign : jnp.ndarray
        The sign of the edges.
    edge_index : jnp.ndarray
        The edge index of the graph.
    """

    position_i = state.position[edge_index[0]]
    position_j = state.position[edge_index[1]]

    edge_forces = compute_force(params, position_i, position_j, sign)
    node_forces = jnp.zeros_like(state.position)
    node_forces = node_forces.at[edge_index[0]].add(edge_forces)

    velocity = state.velocity + 0.5 * params.time_step * node_forces
    position = state.position + params.time_step * velocity

    edge_forces = compute_force(params, position_i, position_j, sign)
    node_forces = jnp.zeros_like(state.position)
    node_forces = node_forces.at[edge_index[0]].add(edge_forces)

    velocity = velocity + 0.5 * params.time_step * node_forces

    velocity = velocity * (1.0 - params.damping)

    energy = jnp.sum(jnp.square(velocity), axis=1)

    return SpringState(position, velocity, energy)

class SimulationParams(NamedTuple):
    iterations : int
    edge_index : jnp.ndarray
    signs : jnp.ndarray

@partial(jax.jit, static_argnums=(1, 2))
def simulate(
    spring_state : SpringState,
    spring_params : SpringParams,
    simulation_params : SimulationParams,
    ) -> SpringState:

    spring_state =jax.lax.fori_loop(
        0, 
        simulation_params.iterations, 
        lambda i, 
        state: update(spring_state, spring_params, simulation_params.signs, simulation_params.edge_index), 
        spring_state)

    embeddings = spring_state.position
    position_i = embeddings.at[simulation_params.edge_index[0]].get()
    position_j = embeddings.at[simulation_params.edge_index[1]].get()

    spring_vec = position_i - position_j
    spring_vec_norm = jnp.linalg.norm(spring_vec, axis=1) - spring_params.neutral_distance

    sigmoid = lambda x: 1.0 / (1.0 + jnp.exp(-x))

    return jnp.sum(sigmoid(spring_vec_norm) - simulation_params.signs)