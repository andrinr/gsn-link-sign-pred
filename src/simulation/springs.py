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

    # if not use_neural_force:
    #     # mass degree correction
    #     node_accelerations = node_accelerations / ( 1 + force_params.mass_degree_correction * graph.node_degrees)

    velocity = spring_state.velocity * (1 - simulation_params.damping) + simulation_params.dt * node_accelerations
    
    # if not use_neural_force:
    #     # center attractor
    #     velocity += spring_state.position * force_params.center_attraction * simulation_params.dt * 0.1

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
        force = neural_force(params, distance, degs_i, degs_j, graph.sign_one_hot)
    else:
        force = heuristic_force(params, distance, graph.sign)

    return force * spring_vector_norm

def heuristic_force(
    params : sm.HeuristicForceParams,
    distance : jnp.ndarray,
    sign : jnp.ndarray
) -> jnp.ndarray:
    
    friend_force = jnp.where(distance < params.friend_segment[0],
                            (distance - params.friend_segment[0]) * params.friend_slope[0] +\
                            params.friend_intercept, 
                            0)

    friend_force = jnp.where((distance >= params.friend_segment[0]) & (distance < params.friend_segment[1]),
                            (distance - params.friend_segment[0]) * params.friend_slope[1] +\
                            params.friend_intercept, friend_force)
    
    friend_force = jnp.where(distance >= params.friend_segment[1],
                            (distance - params.friend_segment[1]) * params.friend_slope[2] +\
                            params.friend_intercept + (params.friend_segment[1] - params.friend_segment[0]) * params.friend_slope[1],
                            friend_force)
    
    enemy_force = jnp.where(distance < params.enemy_segment[0],
                            (distance - params.enemy_segment[0]) * params.enemy_slope[0] +\
                            params.enemy_intercept, 
                            0)
    
    enemy_force = jnp.where((distance >= params.enemy_segment[0]) & (distance < params.enemy_segment[1]),
                            (distance - params.enemy_segment[0]) * params.enemy_slope[1] +\
                            params.enemy_intercept, enemy_force)
    
    enemy_force = jnp.where(distance >= params.enemy_segment[1],
                            (distance - params.enemy_segment[1]) * params.enemy_slope[2] +\
                            params.enemy_intercept + (params.enemy_segment[1] - params.enemy_segment[0]) * params.enemy_slope[1],
                            enemy_force)
    
    neutral_force = jnp.where(distance < params.neutral_segment[0],
                            (distance - params.neutral_segment[0]) * params.neutral_slope[0] +\
                            params.neutral_intercept, 
                            0)
    
    neutral_force = jnp.where((distance >= params.neutral_segment[0]) & (distance < params.neutral_segment[1]),
                            (distance - params.neutral_segment[0]) * params.neutral_slope[1] +\
                            params.neutral_intercept, neutral_force)
    
    neutral_force = jnp.where(distance >= params.neutral_segment[1],
                            (distance - params.neutral_segment[1]) * params.neutral_slope[2] +\
                            params.neutral_intercept + (params.neutral_segment[1] - params.neutral_segment[0]) * params.neutral_slope[1],
                            neutral_force)
    
  
    sign = jnp.expand_dims(sign, axis=1)

    force = jnp.where(sign == 1, friend_force, enemy_force)
    force = jnp.where(sign == 0, neutral_force, force)

    return force

def neural_force(
    params : sm.NeuralForceParams,
    distance : jnp.ndarray,
    degs_i : jnp.ndarray,
    degs_j : jnp.ndarray,
    sign_one_hot : jnp.ndarray,
) -> jnp.ndarray:

    x = jnp.concatenate([sign_one_hot, degs_i, degs_j, distance], axis=1)

    force = sm.mlp_forces(x, params)

    return force * 50