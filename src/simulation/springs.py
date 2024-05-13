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

    position = spring_state.position + simulation_params.dt * node_accelerations

    spring_state = spring_state._replace(position=position)
    
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

    degree_in_i = graph.degree_in.values[graph.edge_index[0]]
    degree_out_i = graph.degree_out.values[graph.edge_index[0]]
    degree_in_j = graph.degree_in.values[graph.edge_index[1]]
    degree_out_j = graph.degree_out.values[graph.edge_index[1]]

    degree_in_neg_i = graph.neg_degree_in.values[graph.edge_index[0]]
    degree_out_neg_i = graph.neg_degree_out.values[graph.edge_index[0]]
    degree_in_neg_j = graph.neg_degree_in.values[graph.edge_index[1]]
    degree_out_neg_j = graph.neg_degree_out.values[graph.edge_index[1]]

    degree_in_pos_i = graph.pos_degree_in.values[graph.edge_index[0]]
    degree_out_pos_i = graph.pos_degree_out.values[graph.edge_index[0]]
    degree_in_pos_j = graph.pos_degree_in.values[graph.edge_index[1]]
    degree_out_pos_j = graph.pos_degree_out.values[graph.edge_index[1]]

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + EPSILON)

    input_i = jnp.concatenate([
        degree_in_i, degree_out_i,
        degree_in_neg_i, degree_out_neg_i,
        degree_in_pos_i, degree_out_pos_i,
        degree_in_j, degree_out_j,
        degree_in_neg_j, degree_out_neg_j,
        degree_in_pos_j, degree_out_pos_j,
        distance], axis=1)

    friend = sm.evaluate_mlp(params.friend, input_i)
    neutral = sm.evaluate_mlp(params.neutral, input_i)
    enemy = sm.evaluate_mlp(params.enemy, input_i)

    sign = jnp.expand_dims(graph.sign, axis=1)

    per_edge_force = jnp.where(sign == 1, friend, enemy)
    per_edge_force = jnp.where(sign == 0, neutral, per_edge_force)

    per_edge_force_i = per_edge_force * spring_vector_norm
    per_edge_force_j = per_edge_force * spring_vector_norm * -1

    per_node_force = jnp.zeros_like(state.position)
    per_node_force = per_node_force.at[graph.edge_index[0]].add(per_edge_force_i)
    per_node_force = per_node_force.at[graph.edge_index[1]].add(per_edge_force_j)

    # mass is constant for all nodes
    per_node_acceleration = per_node_force
    return per_node_acceleration