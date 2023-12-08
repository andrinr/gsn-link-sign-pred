import jax.numpy as jnp

import graph as g
import simulation as sm
EPSILON = 1e-6

def update_spring_state(
    simulation_params : sm.SimulationParams,
    force_params : sm.HeuristicForceParams | sm.NeuralForceParams,
    use_neural_force : bool,
    spring_state : sm.SpringState, 
    graph : g.SignedGraph,
) -> sm.SpringState:

    edge_acceleration = acceleration(force_params, use_neural_force, spring_state, graph)

    node_accelerations = jnp.zeros_like(spring_state.position)
    node_accelerations = node_accelerations.at[graph.edge_index[0]].add(edge_acceleration)
    factor = (force_params.degree_multiplier * jnp.minimum(graph.node_degrees, graph.percentile_degree) / graph.percentile_degree + 1)
    node_accelerations = node_accelerations * factor

    velocity = spring_state.velocity * (1 - simulation_params.damping)
    velocity = velocity + simulation_params.dt * node_accelerations

    velocity_magnitude = jnp.linalg.norm(velocity, axis=1, keepdims=True)
    # limit the velocity to a maximum value
    velocity = velocity * jnp.minimum(velocity_magnitude, 30) / (velocity_magnitude + 1.0)
    
    position = spring_state.position + simulation_params.dt * velocity

    spring_state = spring_state._replace(velocity=velocity, position=position)
    
    return spring_state
  
def acceleration(
    params : sm.HeuristicForceParams | sm.NeuralForceParams,
    use_neural_force : bool,
    state : sm.SpringState,
    graph : g.SignedGraph
) -> jnp.ndarray:
    
    position_i = state.position[graph.edge_index[0]]
    position_j = state.position[graph.edge_index[1]]
    degs_i = graph.node_degrees[graph.edge_index[0]]
    degs_j = graph.node_degrees[graph.edge_index[1]]

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + EPSILON)

    if use_neural_force:
        force = neural_force(params, 
                             distance, 
                             position_i,
                             position_j,
                             state.velocity[graph.edge_index[0]],
                             state.velocity[graph.edge_index[1]],
                             degs_i, 
                             degs_j, 
                             graph)
    else:
        force = heuristic_force(params, distance, graph.sign)

    return force * spring_vector_norm

def heuristic_force(
    params : sm.HeuristicForceParams,
    distance : jnp.ndarray,
    sign : jnp.ndarray
) -> jnp.ndarray:
    
    attraction = jnp.maximum(distance - params.friend_distance, 0) * params.friend_stiffness
    neutral = (distance - params.neutral_distance) * params.neutral_stiffness
    retraction = -jnp.maximum(params.enemy_distance - distance, 0) * params.enemy_stiffness

    sign = jnp.expand_dims(sign, axis=1)

    force = jnp.where(sign == 1, attraction, neutral)
    force = jnp.where(sign == -1, retraction, force)

    return force

def neural_force(
    params : sm.NeuralForceParams,
    distance : jnp.ndarray,
    pos_i : jnp.ndarray,
    pos_j : jnp.ndarray,
    vel_i : jnp.ndarray,
    vel_j : jnp.ndarray,
    degs_i : jnp.ndarray,
    degs_j : jnp.ndarray,
    graph : g.SignedGraph
) -> jnp.ndarray:

    x = jnp.concatenate([
        graph.sign_one_hot,
        distance,
        pos_i,
        pos_j,
        vel_i,
        vel_j,
        degs_i, 
        degs_j], axis=1)

    force = sm.mlp_forces(x, params)

    return force * 50