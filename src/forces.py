import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import simulation as sm

key = jax.random.PRNGKey(0)

distances = jnp.linspace(0, 10, 100)
distances = jnp.expand_dims(distances, axis=1)

params = sm.NeuralForceParams(
    friend_distance=2,
    friend_stiffness=1,
    neutral_distance=5,
    neutral_stiffness=0.1,
    enemy_distance=8,
    enemy_stiffness=1,
    degree_multiplier=0)

forces_friend = sm.neural_force(params, distances, jnp.ones((100,)))
forces_neutral = sm.neural_force(params, distances, jnp.zeros((100,)))
forces_enemy = sm.neural_force(params, distances, -jnp.ones((100,)))

plt.plot(distances, forces_friend, label='positive', color='green')
plt.plot(distances, forces_neutral, label='neutral', color='black')
plt.plot(distances, forces_enemy, label='negative', color='red')

plt.xlabel('distance')
plt.ylabel('force')

plt.legend()
plt.show()
