import jax.numpy as jnp
import jax
from typing import NamedTuple

class LogRegState(NamedTuple):
    location: jnp.ndarray
    scale : jnp.ndarray

def init_log_reg_state() -> LogRegState:
    return LogRegState(jnp.zeros(1), jnp.zeros(1))

jax.jit
def loss(state : LogRegState, X : jnp.ndarray, y : jnp.ndarray) -> float:
    prob = 1.0 / (1.0 + jnp.exp(-(X - state.location) / state.scale))

    return -jnp.sum(y * jnp.log(prob) + (1 - y) * jnp.log(1 - prob))

@jax.jit
def train(state : LogRegState, rate : float, X : jnp.array, y : jnp.ndarray) -> LogRegState:
    
    value, grad = jax.value_and_grad(loss, argnums=0)(state, X, y)

    new_state = LogRegState(state.location - rate * grad[0], state.scale - rate * grad[1])

    return new_state, value

@jax.jit
def predict(state : LogRegState, X : jnp.ndarray) -> jnp.ndarray:
    prob = 1.0 / (1.0 + jnp.exp(-(X - state.location) / state.scale))

    return prob > 0.5