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
    use_neural_force : bool,
    force_params : sm.NeuralForceParams | sm.SpringForceParams,
    spring_state : sm.SpringState, 
    graph : g.SignedGraph,
) -> sm.SpringState:

    if use_neural_force:
        node_accelerations = neural_node_acceleration(force_params, spring_state, graph)
    else:
        node_accelerations = spring_node_acceleration(force_params, spring_state, graph)

    velocity = spring_state.velocity * (1 - simulation_params.damping)
    velocity = velocity + simulation_params.dt * node_accelerations

    position = spring_state.position + simulation_params.dt * velocity

    spring_state = spring_state._replace(velocity=velocity, position=position)
    
    return spring_state
  
def spring_node_acceleration(
    params : sm.SpringForceParams,
    state : sm.SpringState,
    graph : g.SignedGraph) -> jnp.ndarray:

    position_i = state.position[graph.edge_index[0]]
    position_j = state.position[graph.edge_index[1]]

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + EPSILON)

    attraction = jnp.maximum(distance - params.friend_distance, 0) * params.friend_stiffness /2
    neutral = (distance - params.neutral_distance) * params.neutral_stiffness /2
    retraction = -jnp.maximum(params.enemy_distance - distance, 0) * params.enemy_stiffness /2
   
    sign = jnp.expand_dims(graph.sign, axis=1)

    per_edge_force = jnp.where(sign == 1, attraction, neutral)
    per_edge_force = jnp.where(sign == -1, retraction, per_edge_force)
    per_edge_force *= spring_vector_norm

    per_node_force = jnp.zeros_like(state.position)
    per_node_force = per_node_force.at[graph.edge_index[0]].add(per_edge_force)
    per_node_acceleration = per_node_force * (graph.centrality.values * params.degree_multiplier + 0.1)    

    return per_node_acceleration

def neural_node_acceleration(
    params : sm.NeuralForceParams,
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

    direction = jnp.zeros((len(graph.edge_index[0]), 1))

    input_i = jnp.concatenate([
        degree_i, degree_j, 
        degree_i_neg, degree_j_neg, 
        degree_i_pos, degree_j_pos,
        distance, direction], axis=1)

    friend_i = jnp.dot(input_i, params.friend.w0) + params.friend.b0
    friend_i = jnp.dot(friend_i, params.friend.w1) + params.friend.b1

    neutral_i = jnp.dot(input_i, params.neutral.w0) + params.neutral.b0
    neutral_i = jnp.dot(neutral_i, params.neutral.w1) + params.neutral.b1
  
    enemy_i = jnp.dot(input_i, params.enemy.w0) + params.enemy.b0
    enemy_i = jnp.dot(enemy_i, params.enemy.w1) + params.enemy.b1

    direction = jnp.ones((len(graph.edge_index[0]), 1))
    
    # input_j = jnp.concatenate([
    #     degree_j, degree_i, 
    #     degree_j_neg, degree_i_neg, 
    #     degree_j_pos, degree_i_pos,
    #     distance, direction], axis=1)

    # friend_j = jnp.dot(input_j, params.friend.w0) + params.friend.b0
    # friend_j = jnp.dot(friend_j, params.friend.w1) + params.friend.b1

    # neutral_j = jnp.dot(input_j, params.neutral.w0) + params.neutral.b0
    # neutral_j = jnp.dot(neutral_j, params.neutral.w1) + params.neutral.b1

    # enemy_j = jnp.dot(input_j, params.enemy.w0) + params.enemy.b0
    # enemy_j = jnp.dot(enemy_j, params.enemy.w1) + params.enemy.b1

    sign = jnp.expand_dims(graph.sign, axis=1)

    per_edge_force_i = jnp.where(sign == 1, friend_i, enemy_i)
    per_edge_force_i = jnp.where(sign == 0, neutral_i, per_edge_force_i)
    per_edge_force_i *= spring_vector_norm

    # per_edge_force_j = jnp.where(sign == 1, friend_j, enemy_j)
    # per_edge_force_j = jnp.where(sign == 0, neutral_j, per_edge_force_j)
    # per_edge_force_j *= spring_vector_norm

    per_node_force = jnp.zeros_like(state.position)
    per_node_force = per_node_force.at[graph.edge_index[0]].add(per_edge_force_i)
    # per_node_force = per_node_force.at[graph.edge_index[1]].add(per_edge_force_j)

    # mass is constant for all nodes
    per_node_acceleration = per_node_force
    return per_node_acceleration