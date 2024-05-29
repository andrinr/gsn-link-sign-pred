import jax.numpy as jnp
import jax
import graph as g
import training as train
EPSILON = 1e-6

# differntiable version of clip
@jax.jit
def min(
    x : jnp.ndarray,
    min : float,
) -> jnp.ndarray:
    return jnp.where(x < min, min, x)

def update_spring_state(
    simulation_params : train.SimulationParams,
    use_neural_force : bool,
    force_params : train.NeuralForceParams | train.SpringForceParams,
    spring_state : train.SpringState, 
    graph : g.SignedGraph,
) -> train.SpringState:

    # Neural uses 1st order neural ODE
    # if True: #use_neural_force:
    if use_neural_force:
        node_acceleration = neural_node_acceleration(force_params, spring_state, graph)
    else:
        node_acceleration = spring_node_acceleration(force_params, spring_state, graph)

    node_velocity = spring_state.velocity * (1 - simulation_params.damping)
    node_velocity = node_velocity + simulation_params.dt * node_acceleration

    position = spring_state.position + simulation_params.dt * node_velocity

    spring_state = spring_state._replace(position=position)
    
    return spring_state

def spring_node_acceleration(
    params : train.SpringForceParams,
    state : train.SpringState,
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
    params : train.NeuralForceParams,
    state : train.SpringState,
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

    input = jnp.concatenate([
        degree_i, degree_j, 
        degree_i_neg, degree_j_neg, 
        degree_i_pos, degree_j_pos,
        distance], axis=1)

    friend = train.evaluate_mlp(params.friend, input)

    neutral = train.evaluate_mlp(params.neutral, input)

    enemy = train.evaluate_mlp(params.enemy, input)

    sign = jnp.expand_dims(graph.sign, axis=1)

    per_edge_force = jnp.where(sign == 1, friend, enemy)
    per_edge_force = jnp.where(sign == 0, neutral, per_edge_force)
    per_edge_force *= spring_vector_norm

    per_node_force = jnp.zeros_like(state.position)
    per_node_force = per_node_force.at[graph.edge_index[0]].add(per_edge_force)

    # mass is constant for all nodes
    per_node_acceleration = per_node_force
    return per_node_acceleration