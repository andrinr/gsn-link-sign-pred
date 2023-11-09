import jax.numpy as jnp
import jax
import neural as nn
import simulation as sim
from helpers import SignedGraph

EPSILON = 1e-6

# @partial(jax.jit, static_argnames=["nn_force"])
def force_decision(
    spring_state : sim.SpringState,
    nn_force : bool,
    nn_force_params : dict,
    graph : SignedGraph) -> jnp.ndarray:
    
    sign_one_hot = jax.nn.one_hot(graph.sign + 1, 3)

    if not nn_force:
        return spring_state._replace(
            force_decision = sign_one_hot
        )

    auxillaries_i = spring_state.auxillary[graph.edge_index[0]]
    auxillaries_j = spring_state.auxillary[graph.edge_index[1]]

    decision = nn.mlp_forces(
        jnp.concatenate([auxillaries_i, auxillaries_j, sign_one_hot], axis=-1),
        nn_force_params)

    return spring_state._replace(
        force_decision = decision
    )

# @partial(jax.jit, static_argnames=[])
def compute_acceleration(
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

    acceleration = jnp.squeeze(attraction) * state.force_decision[:,2]
    acceleration += jnp.squeeze(neutral) * state.force_decision[:,1]
    acceleration += jnp.squeeze(retraction) * state.force_decision[:,0]

    acceleration = jnp.expand_dims(acceleration, axis=-1)
    
    return acceleration * spring_vector_norm

# @partial(jax.jit, static_argnames=["simulation_params"])
def update_spring_state(
    simulation_params : sim.SimulationParams,
    spring_params : sim.SpringParams,
    spring_state : sim.SpringState, 
    graph : SignedGraph) -> sim.SpringState:
    """
    Update the spring state using the leapfrog method. 
    This is essentially a simple message passing network implementation. 
    """
    edge_acceleration = compute_acceleration(
        spring_params,
        spring_state,
        graph)

    node_accererlations = jnp.zeros_like(spring_state.position)
    node_accererlations = node_accererlations.at[graph.edge_index[0]].add(edge_acceleration)

    velocity = spring_state.velocity + 0.5 * simulation_params.dt * node_accererlations
    position = spring_state.position + simulation_params.dt * velocity

    spring_state = spring_state._replace(
        position=position,
        velocity=velocity)

    edge_acceleration = compute_acceleration(
        spring_params,
        spring_state,
        graph)
    
    node_accererlations = jnp.zeros_like(spring_state.position)
    node_accererlations = node_accererlations.at[graph.edge_index[0]].add(edge_acceleration)

    velocity = velocity + 0.5 * simulation_params.dt * node_accererlations
    velocity = velocity * (1.0 - simulation_params.damping)

    spring_state = spring_state._replace(
        position=position,
        velocity=velocity)
    
    return spring_state