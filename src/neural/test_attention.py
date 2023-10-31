import neural as nn
import jax
import jax.numpy as jnp
import optax

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
    
@jax.jit
def loss(params, x, y):
    y_pred = nn.mlp(x, params)

    print(y_pred.shape) 
    print(y.shape)
    
    return jnp.mean((y_pred - y)**2)

def train_mlp(x, y, params, num_epochs):
    optimizer = optax.adam(learning_rate=0.1)
    optimizer_state = optimizer.init(params)

    for epoch in range(num_epochs):
        loss_value, gradient = jax.value_and_grad(loss)(params, x, y)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        
    return params

def test_learn_addition():
    key0, key1 = jax.random.split(jax.random.PRNGKey(0), num=2)
    params = nn.init_mlp_params(
        key=key0,
        layer_dimensions=[2, 1])

    x = jax.random.uniform(shape=(100, 2), key=key1)

    y = x[:, 0] + x[:, 1]
    y = y.reshape(-1, 1)

    x_train = x[:80]
    y_train = y[:80]

    params = train_mlp(x_train, y_train, params, num_epochs=200)

    x_test = x[80:]
    y_test = y[80:]

    y_test_pred = nn.mlp(x_test, params)

    assert jnp.allclose(y_test_pred, y_test, atol=0.01)