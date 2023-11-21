import jax.numpy as jnp
import jax
import neural as nn
import simulation as sim
from graph import SignedGraph
from functools import partial

EPSILON = 1e-6

def nn_based_force(
    spring_state : sim.SpringState,
    nn_force_params : dict,
    graph : SignedGraph) -> jnp.ndarray:

    position_i = spring_state.position[graph.edge_index[0]]
    position_j = spring_state.position[graph.edge_index[1]]

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + EPSILON)

    sign_one_hot = jax.nn.one_hot(graph.sign + 1, 3)
    degree = jnp.expand_dims(graph.node_degrees[graph.edge_index[0]], axis=-1)

    forces = nn.mlp_forces(
        jnp.concatenate([distance, degree, sign_one_hot], axis=-1),
        nn_force_params)
        
    return forces * spring_vector_norm
    
# @partial(jax.jit, static_argnames=[])
def force(
    params : sim.SpringParams,
    state : sim.SpringState,
    graph : SignedGraph) -> jnp.ndarray:

    position_i = state.position[graph.edge_index[0]]
    position_j = state.position[graph.edge_index[1]]

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + EPSILON)

    attraction = jnp.maximum(distance - params.friend_distance, 0) * params.friend_stiffness
    neutral = (distance - params.neutral_distance) * params.neutral_stiffness
    retraction = -jnp.maximum(params.enemy_distance - distance, 0) * params.enemy_stiffness

    sign = jnp.expand_dims(graph.sign, axis=1)

    acceleration = jnp.where(sign == 1, attraction, neutral)
    acceleration = jnp.where(sign == -1, retraction, acceleration)

    acceleration -= params.center_attraction * (position_i - jnp.mean(position_i, axis=0, keepdims=True))

    return acceleration * spring_vector_norm

#@partial(jax.jit, static_argnames=["simulation_params"])
def update_spring_state(
    simulation_params : sim.SimulationParams,
    spring_params : sim.SpringParams,
    spring_state : sim.SpringState, 
    nn_force : bool,
    nn_force_params : dict,
    graph : SignedGraph
    ) -> sim.SpringState:
    """
    Update the spring state using the kick drift kick integratoin scheme. 
    This is essentially a simple message passing network implementation. 
    """

    if nn_force:
        edge_acceleration = nn_based_force(
            spring_state,
            nn_force_params,
            graph)
    else:
        edge_acceleration = force(
            spring_params,
            spring_state,
            graph)
        
    node_accelerations = jnp.zeros_like(spring_state.position)
    node_accelerations = node_accelerations.at[graph.edge_index[0]].add(edge_acceleration)

    velocity = spring_state.velocity * (1 - simulation_params.damping) + simulation_params.dt * node_accelerations

    position = spring_state.position + simulation_params.dt * velocity

    spring_state = spring_state._replace(velocity=velocity, position=position, acceleration=node_accelerations)
    
    return spring_state