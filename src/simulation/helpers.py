import jax.numpy as jnp
import jax

@jax.jit
def f_beta(truth : jnp.array, prediction : jnp.array, beta : float) -> float:
    tp = jnp.sum(prediction * truth)
    fp = jnp.sum(prediction * (1 - truth))
    fn = jnp.sum((1 - prediction) * truth)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

@jax.jit
def f1_macro(truth : jnp.array, prediction : jnp.array) -> float:
    tp = jnp.sum(prediction * truth, axis=0)
    fp = jnp.sum(prediction * (1 - truth), axis=0)
    fn = jnp.sum((1 - prediction) * truth, axis=0)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return jnp.mean(2 * precision * recall / (precision + recall))