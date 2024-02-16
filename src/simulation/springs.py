import jax.numpy as jnp
import jax
import graph as g
import simulation as sm
EPSILON = 1e-6

# differntiable version of clip
@jax.jit
def min(
    x : jnp.ndarray,
    min : float,
) -> jnp.ndarray:
    return jnp.where(x < min, min, x)

def update_spring_state(
    simulation_params : sm.SimulationParams,
    force_params : sm.HeuristicForceParams,
    spring_state : sm.SpringState, 
    graph : g.SignedGraph,
) -> sm.SpringState:

    edge_acceleration = acceleration(force_params, spring_state, graph)

    node_accelerations = jnp.zeros_like(spring_state.position)
    node_accelerations = node_accelerations.at[graph.edge_index[0]].add(edge_acceleration)
    # node_accelerations = node_accelerations * (graph.centrality.values * force_params.degree_multiplier + 0.1)
    # node_accelerations = spring_state.position * -simulation_params.centering + node_accelerations

    velocity = spring_state.velocity * (1 - simulation_params.damping)
    velocity = velocity + simulation_params.dt * node_accelerations

    # velocity_magnitude = jnp.linalg.norm(velocity, axis=1, keepdims=True)
    # # limit the velocity to a maximum value
    # velocity = velocity * jnp.minimum(velocity_magnitude, 20) / (velocity_magnitude + 1.0)
    
    position = spring_state.position + simulation_params.dt * velocity

    spring_state = spring_state._replace(velocity=velocity, position=position)
    
    return spring_state
  
def acceleration(
    params : sm.HeuristicForceParams,
    state : sm.SpringState,
    graph : g.SignedGraph
) -> jnp.ndarray:
    
    position_i = state.position[graph.edge_index[0]]
    position_j = state.position[graph.edge_index[1]]

    degree_i = graph.degree.values[graph.edge_index[0]]
    degree_j = graph.degree.values[graph.edge_index[1]]
    degree_i_neg = graph.neg_degree.values[graph.edge_index[0]]
    degree_j_neg = graph.neg_degree.values[graph.edge_index[1]]
    degree_i_pos = graph.pos_degree.values[graph.edge_index[0]]
    degree_j_pos = graph.pos_degree.values[graph.edge_index[1]]

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + EPSILON)

    force_factor = heuristic_force(
        params, 
        degree_i, 
        degree_j, 
        degree_i_neg, 
        degree_j_neg, 
        degree_i_pos,
        degree_j_pos,
        distance, 
        graph.sign)

    return force_factor * spring_vector_norm

def heuristic_force(
    params : sm.HeuristicForceParams,
    degree_i : jnp.ndarray,
    degree_j : jnp.ndarray,
    degree_negative_i : jnp.ndarray,
    degree_negative_j : jnp.ndarray,
    degree_positive_i : jnp.ndarray,
    degree_positive_j : jnp.ndarray,
    distance : jnp.ndarray,
    sign : jnp.ndarray
) -> jnp.ndarray:
    
    input = jnp.concatenate([
        degree_i, degree_j, 
        degree_negative_i, degree_negative_j, 
        degree_positive_i, degree_positive_j,
        distance], axis=1)

    friend = jnp.dot(input, params.friend.w0) + params.friend.b0
    # friend = jax.nn.relu(friend)
    friend = jnp.dot(friend, params.friend.w1) + params.friend.b1

    neutral = jnp.dot(input, params.neutral.w0) + params.neutral.b0
    # neutral = jax.nn.relu(neutral)
    neutral = jnp.dot(neutral, params.neutral.w1) + params.neutral.b1
  
    enemy = jnp.dot(input, params.enemy.w0) + params.enemy.b0
    # enemy = jax.nn.relu(enemy)
    enemy = jnp.dot(enemy, params.enemy.w1) + params.enemy.b1

    # friend = jnp.maximum(distance - params.friend.rest_length, 0) * \
    #     params.friend.attraction_stiffness * (1 + params.friend.degree_i_multiplier * degree_i) * (1 + params.friend.degree_j_multiplier * degree_j)
    # friend += jnp.minimum(distance - params.friend.rest_length, 0) *\
    #       params.friend.repulsion_stiffness * (1 + params.friend.degree_i_multiplier * degree_i) * (1 + params.friend.degree_j_multiplier * degree_j)

    # neutral = jnp.maximum(distance - params.neutral.rest_length, 0) * \
    #     params.neutral.attraction_stiffness * (1 + params.neutral.degree_i_multiplier * degree_i) * (1 + params.neutral.degree_j_multiplier * degree_j)
    # neutral += jnp.minimum(distance - params.neutral.rest_length, 0) * \
    #     params.neutral.repulsion_stiffness * (1 + params.neutral.degree_i_multiplier * degree_i) * (1 + params.neutral.degree_j_multiplier * degree_j)

    # enemy = jnp.maximum(distance - params.enemy.rest_length, 0) * \
    #     params.enemy.attraction_stiffness * (1 + params.enemy.degree_i_multiplier * degree_i) * (1 + params.enemy.degree_j_multiplier * degree_j)
    # enemy += jnp.minimum(distance - params.enemy.rest_length, 0) * \
    #     params.enemy.repulsion_stiffness * (1 + params.enemy.degree_i_multiplier * degree_i) * (1 + params.enemy.degree_j_multiplier * degree_j)

    sign = jnp.expand_dims(sign, axis=1)

    force = jnp.where(sign == 1, friend, enemy)
    force = jnp.where(sign == 0, neutral, force)

    return force
