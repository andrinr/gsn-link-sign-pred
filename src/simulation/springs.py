import jax.numpy as jnp
import simulation as sim
from graph import SignedGraph
from neural import NeuralForceParams, mlp_forces

EPSILON = 1e-6
    
# @partial(jax.jit, static_argnames=[])
def heuristic_force(
    params : sim.HeuristicForceParams,
    state : sim.SpringState,
    graph : SignedGraph
) -> jnp.ndarray:
    
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

def neural_force(
    params : NeuralForceParams,
    state : sim.SpringState,
    graph : SignedGraph
) -> jnp.ndarray:
    
    position_i = state.position[graph.edge_index[0]]
    position_j = state.position[graph.edge_index[1]]
    degs_i = graph.node_degrees[graph.edge_index[0]]
    degs_j = graph.node_degrees[graph.edge_index[1]]

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + EPSILON)

    degs_i = jnp.expand_dims(degs_i, axis=1)
    degs_j = jnp.expand_dims(degs_j, axis=1)
    
    x = jnp.concatenate([graph.sign_one_hot, distance, degs_i, degs_j], axis=1)

    x = mlp_forces(x, params)

    return x * 10 * spring_vector_norm

def update_spring_state(
    simulation_params : sim.SimulationParams,
    force_params : sim.HeuristicForceParams | NeuralForceParams,
    use_neural_force : bool,
    spring_state : sim.SpringState, 
    graph : SignedGraph,
) -> sim.SpringState:

    if use_neural_force:
        edge_acceleration = neural_force(
            force_params,
            spring_state,
            graph)
    else:
        edge_acceleration = heuristic_force(
            force_params,
            spring_state,
            graph)
        
    node_accelerations = jnp.zeros_like(spring_state.position)
    node_accelerations = node_accelerations.at[graph.edge_index[0]].add(edge_acceleration)

    velocity = spring_state.velocity * (1 - simulation_params.damping) + simulation_params.dt * node_accelerations

    position = spring_state.position + simulation_params.dt * velocity

    spring_state = spring_state._replace(velocity=velocity, position=position, acceleration=node_accelerations)
    
    return spring_state