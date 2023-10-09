import numpy as np

import jax.numpy as jnp

a = jnp.array([[1, 2, 3], [4, 5, 6]])
b = jnp.array([[2, 3, 4], [6, 5, 6]])

print(a.shape)
print(b.shape)

c = jnp.array([0,1])

c = jnp.expand_dims(c, axis=1)


print(jnp.where(c == 1, a, b))