import jax.numpy as jnp
import jax
import graph as g
import simulation as sm
EPSILON = 1e-6

def init_spring_force_params() -> sm.SpringForceParams:
    return sm.SpringForceParams(
        friend_distance=1.0,
        friend_stiffness=1.0,
        neutral_distance=1.0,
        neutral_stiffness=0.1,
        enemy_distance=5.5,
        enemy_stiffness=2.0,
        degree_multiplier=3.0)

def init_node_state(
    rng : jax.random.PRNGKey, 
    n : int, m : int,
    range : float,
    embedding_dim : int) -> sm.NodeState:
    position = jax.random.uniform(rng, (n, embedding_dim), minval=-range, maxval=range)
    velocity = jnp.zeros((n, embedding_dim))
    acceleration = jnp.zeros((n, embedding_dim))
    return sm.NodeState(position, velocity, acceleration)

# differntiable version of clip
@jax.jit
def min(
    x : jnp.ndarray,
    min : float,
) -> jnp.ndarray:
    return jnp.where(x < min, min, x)

def update(
    simulation_params : sm.SimulationParams,
    use_neural_force : bool,
    force_params : sm.NeuralEdgeParams | sm.SpringForceParams,
    node_state : sm.NodeState, 
    graph : g.SignedGraph,
) -> sm.NodeState:

    if use_neural_force:
        node_accelerations = \
            neural_node_acceleration(force_params, node_state, graph) *\
            neural_node_scaling(force_params, graph)
    else:
        node_accelerations =\
            spring_node_acceleration(force_params, node_state, graph) *\
            spring_node_scaling(force_params, graph)

    velocity = node_state.velocity * (1 - simulation_params.damping)
    velocity = velocity + simulation_params.dt * node_accelerations

    position = node_state.position + simulation_params.dt * velocity

    node_state = node_state._replace(
        velocity=velocity, 
        position=position, 
        acceleration=node_accelerations)
    
    return node_state

def spring_node_scaling(
    params : sm.SpringForceParams,
    graph : g.SignedGraph,
) -> jnp.ndarray:
    return (graph.centrality.values * params.degree_multiplier + 0.1)    

def spring_node_acceleration(
    params : sm.SpringForceParams,
    node_state : sm.NodeState,
    graph : g.SignedGraph) -> jnp.ndarray:

    position_i = node_state.position[graph.edge_index[0]]
    position_j = node_state.position[graph.edge_index[1]]

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

    per_node_force = jnp.zeros_like(node_state.position)
    per_node_force = per_node_force.at[graph.edge_index[0]].add(per_edge_force)

    return per_node_force

def neural_node_scaling(
    params : sm.NeuralParams,
    graph : g.SignedGraph,
) -> jnp.ndarray:

    input = jnp.concatenate([
        graph.degree.values,
        graph.out_deg.values,
        graph.in_deg.values,
        graph.out_neg.values + graph.in_neg.values,
        graph.out_pos.values + graph.in_pos.values
        ], axis=1)
    
    return 1 + sm.apply_mlp2(params.node_params, input)

def neural_node_acceleration(
    params : sm.NeuralParams,
    node_state : sm.NodeState,
    graph : g.SignedGraph
) -> jnp.ndarray:
    
    position_i = node_state.position[graph.edge_index[0]]
    position_j = node_state.position[graph.edge_index[1]]

    degree_i = graph.degree.values[graph.edge_index[0]]
    degree_j = graph.degree.values[graph.edge_index[1]]
    degree_i_out = graph.out_deg.values[graph.edge_index[0]]
    degree_j_out = graph.out_deg.values[graph.edge_index[1]]
    degree_i_in = graph.in_deg.values[graph.edge_index[0]]
    degree_j_in = graph.in_deg.values[graph.edge_index[1]]
    degree_i_out_neg = graph.out_neg.values[graph.edge_index[0]]
    degree_j_out_neg = graph.out_neg.values[graph.edge_index[1]]
    degree_i_out_pos = graph.out_pos.values[graph.edge_index[0]]
    degree_j_out_pos = graph.out_pos.values[graph.edge_index[1]]
    degree_i_in_neg = graph.in_neg.values[graph.edge_index[0]]
    degree_j_in_neg = graph.in_neg.values[graph.edge_index[1]]
    degree_i_in_pos = graph.in_pos.values[graph.edge_index[0]]
    degree_j_in_pos = graph.in_pos.values[graph.edge_index[1]]

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + EPSILON)

    input = jnp.concatenate([
        degree_i, degree_j, 
        degree_i_out, degree_j_out,
        degree_i_in, degree_j_in,
        degree_i_out_neg, degree_j_out_neg,
        degree_i_out_pos, degree_j_out_pos,
        degree_i_in_neg, degree_j_in_neg,
        degree_i_in_pos, degree_j_in_pos,
        distance], axis=1)
    
    friend_in = sm.apply_mlp2(params.edge_params.friend_in, input)
    friend_out = sm.apply_mlp2(params.edge_params.friend_out, input)

    neutral_in = sm.apply_mlp2(params.edge_params.neutral_in, input)
    neutral_out = sm.apply_mlp2(params.edge_params.neutral_out, input)

    enemy_in = sm.apply_mlp2(params.edge_params.enemy_in, input)
    enemy_out = sm.apply_mlp2(params.edge_params.enemy_out, input)

    sign = jnp.expand_dims(graph.sign, axis=1)

    edge_in_force = jnp.where(sign == 1, friend_in, enemy_in)
    edge_in_force = jnp.where(sign == 0, neutral_in, edge_in_force)
    edge_in_force *= spring_vector_norm

    edge_out_force = jnp.where(sign == 1, friend_out, enemy_out)
    edge_out_force = jnp.where(sign == 0, neutral_out, edge_out_force)
    edge_out_force *= -spring_vector_norm

    per_node_force = jnp.zeros_like(node_state.position)
    per_node_force = per_node_force.at[graph.edge_index[0]].add(edge_out_force)
    per_node_force = per_node_force.at[graph.edge_index[1]].add(edge_in_force)

    # mass is constant for all nodes
    per_node_acceleration = per_node_force
    return per_node_acceleration