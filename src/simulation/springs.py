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
    factor = (force_params.degree_multiplier * \
              min(graph.centrality.values, graph.centrality.percentile) \
                / graph.centrality.percentile + 1)
    node_accelerations = node_accelerations * factor

    velocity = spring_state.velocity * (1 - simulation_params.damping)
    velocity = velocity + simulation_params.dt * node_accelerations

    velocity_magnitude = jnp.linalg.norm(velocity, axis=1, keepdims=True)
    # limit the velocity to a maximum value
    velocity = velocity * min(velocity_magnitude, 20) / (velocity_magnitude + 1.0)
    
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

    spring_vector = position_j - position_i
    distance = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (distance + EPSILON)

    force = heuristic_force(params, distance, graph.sign)

    return force * spring_vector_norm

def heuristic_force(
    params : sm.HeuristicForceParams,
    distance : jnp.ndarray,
    sign : jnp.ndarray
) -> jnp.ndarray:
    
    attraction = jnp.maximum(distance - params.friend_distance, 0) * params.friend_stiffness /2
    neutral = (distance - params.neutral_distance) * params.neutral_stiffness /2
    retraction = -jnp.maximum(params.enemy_distance - distance, 0) * params.enemy_stiffness /2

    sign = jnp.expand_dims(sign, axis=1)

    force = jnp.where(sign == 1, attraction, neutral)
    force = jnp.where(sign == -1, retraction, force)

    return force
