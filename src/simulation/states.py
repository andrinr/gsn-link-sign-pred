from typing import NamedTuple
import jax.numpy as jnp

class SimulationState(NamedTuple):
    iteration : int
    time : float

class SimulationParams(NamedTuple):
    iterations : int
    dt : float
    damping : float

class MLP2(NamedTuple):
    w0 : jnp.ndarray
    w1 : jnp.ndarray
    b0 : jnp.ndarray
    b1 : jnp.ndarray

class NeuralEdgeParams(NamedTuple):
    friend : MLP2
    neutral : MLP2
    enemy : MLP2

class NeuralParams(NamedTuple):
    edge_params : NeuralEdgeParams
    node_params : MLP2

class NodeState(NamedTuple):
    position: jnp.ndarray
    velocity: jnp.ndarray
    acceleration: jnp.ndarray

class SpringForceParams(NamedTuple):
    friend_distance: float
    friend_stiffness: float
    neutral_distance: float
    neutral_stiffness: float
    enemy_distance: float
    enemy_stiffness: float
    degree_multiplier: float

class Metrics(NamedTuple):
    auc : float
    f1_binary : float
    f1_micro : float
    f1_macro : float
    true_positives : float
    false_positives : float
    true_negatives : float
    false_negatives : float

    def __str__(self):
        return "auc: {self.auc}," + \
        "f1_binary: {self.f1_binary}, " + \
        "f1_micro: {self.f1_micro}, " + \
        "f1_macro: {self.f1_macro}, " + \
        "true_positives: {self.true_positives}, " + \
        "false_positives: {self.false_positives}, " + \
        "true_negatives: {self.true_negatives}, " + \
        "false_negatives: {self.false_negatives}"