import neural as nn
import jax
import jax.numpy as jnp

def test_attention():

    key = jax.random.PRNGKey(0)
    key0, key1, key2 = jax.random.split(key, num=3)
    params = nn.init_attention_params(
        key=key0,
        input_dimension=3,
        output_dimension=4)
    
    x_i = jax.random.normal(shape=(100, 3), key=key1)
    x_j = jax.random.normal(shape=(100, 3), key=key2)

    y = nn.attention(x_i, x_j, params)

    assert jnp.isnan(y).sum() == 0
    
