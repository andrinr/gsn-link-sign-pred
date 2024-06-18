from typing import NamedTuple
import jax.numpy as jnp
import jax
from jax import random
import simulation as sm


def init_neural_params(key : jax.random.PRNGKey
) -> sm.NeuralParams:
    init_orth = jax.nn.initializers.orthogonal()
    init_zeros = jax.nn.initializers.zeros
    init_normal = jax.nn.initializers.normal(stddev=0.1)

    mlps = []

    n_in = 15
    n_hidden = 15
    n_out = 1

    for _ in range(3):
        mlps.append(sm.MLP2(
            w0=init_normal(key, (n_in, n_hidden)),
            w1=init_normal(key, (n_hidden, n_out)),
            b0=init_zeros(key, (n_hidden,)),
            b1=init_zeros(key, (n_out,))))
    
    edge_params = sm.NeuralEdgeParams(
        friend=mlps[0],
        neutral=mlps[1],
        enemy=mlps[2])
    
    n_in = 5
    n_hidden = 3
    n_out = 1
    
    node_params = sm.MLP2(
        w0=init_normal(key, (n_in, n_hidden)),
        w1=init_normal(key, (n_hidden, n_out)),
        b0=init_zeros(key, (n_hidden,)),
        b1=init_zeros(key, (n_out,)))
    
    return sm.NeuralParams(edge_params, node_params)

def apply_mlp2(mlp : sm.MLP2, x : jnp.ndarray) -> jnp.ndarray:
    x = jnp.dot(x, mlp.w0) + mlp.b0
    x = jax.nn.tanh(x)
    x = jnp.dot(x, mlp.w1) + mlp.b1
    return x
