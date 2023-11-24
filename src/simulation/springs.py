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

    velocity = spring_state.velocity * (1 - simulation_params.damping) + simulation_params.dt * node_accelerations

    position = spring_state.position + simulation_params.dt * velocity

    spring_state = spring_state._replace(velocity=velocity, position=position, acceleration=node_accelerations)
    
    return spring_state
  
def acceleration(
    params : sm.HeuristicForceParams | sm.NeuralForceParams,
    use_neural_force : bool,
    state : sm.SpringState,
    graph : g.SignedGraph
) -> jnp.ndarray:
    
    position_i = state.position[graph.edge_index[0]]
    position_j = state.position[graph.edge_index[1]]

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + EPSILON)

    if use_neural_force:
        force = neural_force(params, distance, graph.sign_one_hot)
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
    sign_one_hot : jnp.ndarray,
) -> jnp.ndarray:

    x = jnp.concatenate([sign_one_hot, distance], axis=1)

    force = sm.mlp_forces(x, params)

    return force * 300