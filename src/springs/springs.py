import jax.numpy as jnp
import jax
from typing import NamedTuple

class SpringParams(NamedTuple):
    friend_distance: float
    friend_stiffness: float
    neutral_distance: float
    neutral_stiffness: float
    enemy_distance: float
    enemy_stiffness: float
    damping: float
    time_step: float

class SpringState(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray
    energy: jnp.ndarray

def init_spring_state(rng : jax.random.PRNGKey, n : int, embedding_dim : int) -> SpringState:
    position = jax.random.uniform(rng, (n, embedding_dim))
    velocity = jnp.zeros((n, embedding_dim))
    energy = jnp.zeros(n)

    return SpringState(position, velocity, energy)

@jax.jit
def compute_force(
    params : SpringParams, 
    position_i : jnp.ndarray,
    position_j : jnp.ndarray,
    sign : jnp.ndarray) -> jnp.ndarray:
    
    spring_vector = position_j - position_i
    l = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
    spring_vector_norm = spring_vector / (l + 0.001)

    attraction = jnp.maximum(l - params.friend_distance, 0) * params.friend_stiffness * spring_vector_norm
    neutral = (l - params.neutral_distance) * params.neutral_stiffness * spring_vector_norm
    retraction = -jnp.maximum(params.enemy_distance - l, 0) * params.enemy_stiffness * spring_vector_norm

    sign = jnp.expand_dims(sign, axis=1)
    force = jnp.where(sign == 1, attraction, retraction)
    force = jnp.where(sign == 0, neutral, force)
    
    return force

@jax.jit
def update(
    state : SpringState, 
    params : SpringParams, 
    sign : jnp.ndarray, 
    edge_index : jnp.ndarray) -> SpringState:

    # n = state.position.shape[0]
    # m = edge_index.shape[1]
    # dim = state.position.shape[1]

    position_i = state.position[edge_index[0]]
    position_j = state.position[edge_index[1]]

    edge_forces = compute_force(params, position_i, position_j, sign)
    node_forces = jnp.zeros_like(state.position)
    node_forces = node_forces.at[edge_index[0]].add(edge_forces)

    velocity = state.velocity + 0.5 * params.time_step * node_forces
    position = state.position + params.time_step * velocity

    edge_forces = compute_force(params, position_i, position_j, sign)
    node_forces = jnp.zeros_like(state.position)
    node_forces = node_forces.at[edge_index[0]].add(edge_forces)

    velocity = velocity + 0.5 * params.time_step * node_forces

    velocity = velocity * params.damping

    energy = jnp.sum(jnp.square(velocity), axis=1)

    return SpringState(position, velocity, energy)


#    iterations_interval = self.iterations // num_intervals

#     n = self.train_data.num_nodes
#     m = self.train_data.num_edges

#     key = jnp.random.PRNGKey(0)
#     pos = jnp.random.uniform(key, (n, self.embedding_dim))
#     vel = jnp.zeros((n, self.embedding_dim))
#     signs = self.trainings_signs

#     aucs = jnp.zeros(num_intervals)
#     f1_binaries = jnp.zeros(num_intervals)
#     f1_micros = jnp.zeros(num_intervals)
#     f1_macros = jnp.zeros(num_intervals)

#     for interval in range(num_intervals):
#         for i in range(iterations_interval):

#             pos, vel = self.step(pos, vel, signs, self.edge_index)

#             vel = vel * self.damping

#         pos_i = pos[self.edge_index[0]]
#         pos_j = pos[self.edge_index[1]]

#         y_pred, auc_score, f1_binary, f1_micro, f1_macro = self.log_reg.predict(pos_i, pos_j)

#         aucs[interval] = auc_score
#         f1_binaries[interval] = f1_binary
#         f1_micros[interval] = f1_micro
#         f1_macros[interval] = f1_macro

#     return pos, aucs, f1_binaries, f1_micros, f1_macros
