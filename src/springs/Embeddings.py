import jax.numpy as jnp
from jax import jit

class Embeddings:
    """
    Training
    """
    def __init__(self, 
        edge_index,
        signs,
        signs_truth,
        embedding_dim, 
        time_step, 
        iterations, 
        damping,
        friend_distance,
        friend_stiffness,
        neutral_distance,
        neutral_stiffness,
        enemy_distance,
        enemy_stiffness):

        self.edge_index = edge_index
        self.signs = signs
        self.signs_truth = signs_truth
        self.embedding_dim = embedding_dim
        self.time_step = time_step
        self.iterations = iterations
        self.damping = damping
        self.friend_distance = friend_distance
        self.friend_stiffness = friend_stiffness
        self.neutral_distance = neutral_distance
        self.neutral_stiffness = neutral_stiffness
        self.enemy_distance = enemy_distance
        self.enemy_stiffness = enemy_stiffness

    @jit
    def force(self, pos_i, pos_j, sign):
        spring_vector = pos_j - pos_i
        l = jnp.linalg.norm(spring_vector, axis=1, keepdims=True)
        spring_vector_norm = spring_vector / (l + 0.001)

        attraction = jnp.maximum(l - self.friend_distance, 0) * self.friend_stiffness * spring_vector_norm
        neutral = (l - self.neutral_distance) * self.neutral_stiffness * spring_vector_norm
        retraction = -jnp.maximum(self.enemy_distance - l, 0) * self.enemy_stiffness * spring_vector_norm

        force = jnp.where(sign == 1, attraction, retraction)
        force = jnp.where(sign == 0, neutral, force)
        return force

    @jit
    def step(self, pos, vel, sign, edge_index):

        pos_i = pos[edge_index[0]]
        pos_j = pos[edge_index[1]]

        force = self.force(pos_i, pos_j, sign)

        vel = vel + 0.5 * self.time_step * force
        pos = pos + self.time_step * vel

        force = self.force(pos_i, pos_j, sign)
        vel = vel + 0.5 * self.time_step * force

        return pos, vel

    @jit
    def error(pos_i, pos_j):


    def __call__(self,
        num_intervals : int) -> float:

        iterations_interval = self.iterations // num_intervals

        n = self.train_data.num_nodes
        m = self.train_data.num_edges

        key = jnp.random.PRNGKey(0)
        pos = jnp.random.uniform(key, (n, self.embedding_dim))
        vel = jnp.zeros((n, self.embedding_dim))
        signs = self.train_data.edge_attr

        for i in range(num_intervals):
            for j in range(iterations_interval):

                pos, vel = self.step(pos, vel, signs, self.train_data.edge_index)

                vel = vel * self.damping