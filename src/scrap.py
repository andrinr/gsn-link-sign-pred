import numpy as np

import jax.numpy as jnp


# example matrices
A = np.array([[1, 2], [3, 4], [4, 5], [6, 7]])  # A is 2 x 2 (m x n)
B = np.array([[2, 3], [1, 2]])  # B is 2 x 2 (n x k)

# performing dot product
C = np.dot(A, B)  # C will be 2 x 2 (m x k)

print(C)