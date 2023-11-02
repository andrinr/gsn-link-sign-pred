import numpy as np

import jax.numpy as jnp


# example matrices
A = jnp.array([[1, 0], [0, 1], [4, 5], [10, 7]])  # A is 2 x 2 (m x n)

A = jnp.argmax(A, axis=-1)

B = jnp.array([0, 1, 2, 3])

C = jnp.where(A == 1, 1, 0)

print(C)
