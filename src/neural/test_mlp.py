import neural as nn
import jax
import jax.numpy as jnp

def test_attention():
    key0, key1 = jax.random.split(jax.random.PRNGKey(0), num=2)
    params = nn.init_mlp_params(
        key=key0,
        layer_dimensions=[4, 3, 2])
    
    x = jax.random.normal(shape=(100, 4), key=key1)

    y = nn.mlp(x, params)
    
    assert jnp.isnan(y).sum() == 0
    
