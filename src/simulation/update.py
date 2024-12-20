import jax.numpy as jnp
import jax
import graph as g
import simulation as sm
EPSILON = 1e-6

def init_spring_force_params() -> sm.SpringForceParams:
    return sm.SpringForceParams(
        friend_distance=1.0,
        friend_stiffness=1.0,
        enemy_distance=1.0,
        enemy_stiffness=1.0,
        degree_multiplier=1.0)

def init_node_state(
    rng : jax.random.PRNGKey, 
    n : int,
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
    return (graph.train_centr.values * params.degree_multiplier + 1.0)    

def spring_node_acceleration(
    params : sm.SpringForceParams,
    node_state : sm.NodeState,
    graph : g.SignedGraph) -> jnp.ndarray:

    position_i = node_state.position[graph.train_edge_index[0]]
    position_j = node_state.position[graph.train_edge_index[1]]

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + EPSILON)

    attraction = (distance - params.friend_distance) * params.friend_stiffness /2
    retraction = -(params.enemy_distance - distance) * params.enemy_stiffness /2
   
    sign = jnp.expand_dims(graph.train_sign, axis=1)

    per_edge_force = jnp.where(sign == 1, attraction, retraction)
    per_edge_force *= spring_vector_norm

    per_node_force = jnp.zeros_like(node_state.position)
    per_node_force = per_node_force.at[graph.train_edge_index[0]].add(per_edge_force)

    return per_node_force

def neural_node_scaling(
    params : sm.NeuralParams,
    graph : g.SignedGraph,
) -> jnp.ndarray:

    input = jnp.concatenate([
        graph.train_deg.values,
        graph.train_neg_deg.values,
        graph.train_pos_deg.values], axis=1)
    
    return sm.apply_mlp(params.node_params, input)

def neural_node_acceleration(
    params : sm.NeuralParams,
    node_state : sm.NodeState,
    graph : g.SignedGraph
) -> jnp.ndarray:
    
    position_i = node_state.position[graph.train_edge_index[0]]
    position_j = node_state.position[graph.train_edge_index[1]]

    degree_i = graph.train_deg.values[graph.train_edge_index[0]]
    degree_j = graph.train_deg.values[graph.train_edge_index[1]]
    degree_i_neg = graph.train_neg_deg.values[graph.train_edge_index[0]]
    degree_j_neg = graph.train_neg_deg.values[graph.train_edge_index[1]]
    degree_i_pos = graph.train_pos_deg.values[graph.train_edge_index[0]]
    degree_j_pos = graph.train_pos_deg.values[graph.train_edge_index[1]]

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + EPSILON)

    input = jnp.concatenate([
        degree_i, degree_j, 
        degree_i_neg, degree_j_neg, 
        degree_i_pos, degree_j_pos,
        distance], axis=1)

    friend = sm.apply_mlp(params.edge_params.friend, input)
    enemy = sm.apply_mlp(params.edge_params.enemy, input)

    sign = jnp.expand_dims(graph.train_sign, axis=1)

    per_edge_force = jnp.where(sign == 1, friend, enemy)
    per_edge_force *= spring_vector_norm

    per_node_force = jnp.zeros_like(node_state.position)
    per_node_force = per_node_force.at[graph.train_edge_index[0]].add(per_edge_force)

    # mass is constant for all nodes
    per_node_acceleration = per_node_force
    return per_node_acceleration