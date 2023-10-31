import neural as nn
import jax
import jax.numpy as jnp
import optax

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