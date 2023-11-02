import neural as nn
import jax
import jax.numpy as jnp
import optax
from simulation import update_auxillary_state, SpringState
from functools import partial

# @partial(jax.jit, static_argnames=["iterations"])
def loss(params, spring_state, edge_index, signs, iterations, y):
    for i in range(iterations):
        spring_state = update_auxillary_state(
            spring_state=spring_state,
            auxillaries_nn_params=params,
            edge_index=edge_index,
            sign=signs)
        
        # print(f"spring_state.auxillaries: {spring_state.auxillaries}")
        
    # print(y_pred.shape)
    # print(y.shape)

    return jnp.mean((spring_state.auxillary - y)**2), spring_state
        
def test_bipartition():
    # generate two clusters of points and connect them randomly with edges
    # check if the attention mp algorithm can separate them

    num_nodes_per_cluster = 5
    num_edges = 20

    key = jax.random.PRNGKey(0)
    key0, key1, key2, key3 = jax.random.split(key, num=4)

    edge_i = jax.random.randint(
        key=key0,
        minval=0,
        maxval=num_nodes_per_cluster,
        shape=(num_edges,))
    
    edge_j = jax.random.randint(
        key=key1,
        minval=num_nodes_per_cluster,
        maxval=2*num_nodes_per_cluster,
        shape=(num_edges,))
    
    edge_index = jnp.stack([edge_i, edge_j], axis=1)

    # make undirected
    edge_index = jnp.concatenate([edge_index, edge_index[:, ::-1]], axis=0)

    # flip axes
    edge_index = edge_index.T

    signs = jnp.ones((2*num_edges,))
    y = jnp.concatenate([
        jnp.zeros((num_nodes_per_cluster,)),
        jnp.ones((num_nodes_per_cluster,))], axis=0)
    

    print(edge_index.shape)
    print(signs.shape)
    print(y.shape)

    value_and_grad_fn = jax.value_and_grad(loss, argnums=0, has_aux=True)

    # init the neural network params
    params = nn.init_attention_params(
        key=key2,
        input_dimension=2,
        output_dimension=1,
        factor=0.3)

    print(params)
    
    optimizer = optax.adam(learning_rate=0.01)
    optimizer_state = optimizer.init(params)

    # auxillaries = jax.random.normal(
    #     key=key3,
    #     shape=(2*num_nodes_per_cluster, 2))
    
    # spring_state = SpringState(
    #     position=auxillaries,
    #     velocity=jnp.zeros_like(auxillaries),
    #     auxillaries=auxillaries)
    
    # loss_value, spring_state = loss(
    #     params, spring_state, edge_index, signs, 5, y)
    
    # print(f"solution: {spring_state.auxillaries}")
    num_epochs = 100
    keys = jax.random.split(key3, num=num_epochs)
    for epoch in range(num_epochs):

        print(f" -------- \n epoch: {epoch} \n --------")

        auxillaries = jax.random.normal(
            key=keys[epoch],
            shape=(2*num_nodes_per_cluster, 1))
    
        spring_state = SpringState(
            position=auxillaries,
            velocity=jnp.zeros_like(auxillaries),
            auxillary=auxillaries)
        
        (loss_value, spring_state), gradient = value_and_grad_fn(
            params, spring_state, edge_index, signs, 5, y)

        print(f"loss_value: {loss_value}")
        print(f"gradient: {gradient}")

        updates, optimizer_state = optimizer.update(gradient, optimizer_state, params)
        params = optax.apply_updates(params, updates)

    assert False
