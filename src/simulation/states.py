from typing import NamedTuple
import jax.numpy as jnp

class SpringParams(NamedTuple):
    friend_distance: float
    friend_stiffness: float
    neutral_distance: float
    neutral_stiffness: float
    enemy_distance: float
    enemy_stiffness: float
    distance_threshold: float

class SpringState(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray
    auxillary: jnp.ndarray
    force_decision: jnp.ndarray

class SimulationState(NamedTuple):
    iteration : int
    time : float

class SimulationParams(NamedTuple):
    iterations : int
    dt : float
    damping : float
    message_passing_iterations : int