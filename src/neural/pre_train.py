import jax
import jax.numpy as jnp
from tqdm import tqdm
import optax

# local imports
import simulation as sim
import neural as nn
from graph import SignedGraph

def pre_train(
    key : jax.random.PRNGKey,
    learning_rate : float,
    num_epochs : int,
    heuristic_force_params : nn.NeuralForceParams,
    neural_force_params : nn.NeuralForceParams,
) -> nn.NeuralForceParams:
    
    # generate random SignedGraph
    num_edges = 10000
    num_nodes = 1000

    signs = jax.random.randint(
        key, 
        low=0, 
        high=2, 
        shape=(num_edges,))
    
    edge_index = jax.random.randint(
        key, 
        low=0, 
        high=num_nodes, 
        shape=(2, num_edges))

    signed_graph = SignedGraph(
        edge_index=edge_index,
        sign=signs,
        sign_one_hot=jax.nn.one_hot(signs, 3),
        node_degrees=jax.zeros((num_nodes,)),
        num_nodes=num_nodes,
        num_edges=num_edges,
        train_mask=jnp.zeros((num_nodes,), dtype=bool),
        test_mask=jnp.zeros((num_nodes,), dtype=bool))
    
    spring_state = sim.init_spring_state(
        key,
        range=40,
        n=num_nodes,
        m=num_edges,
        embedding_dim=16)

    # create optimizer
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(neural_force_params)

    # create tqdm progress bar
    epochs = tqdm(range(num_epochs))

    # create value and grad function
    value_and_grad_fn = jax.value_and_grad(pre_train_loss, argnums=1, has_aux=False)

    # pre train
    for epoch_index in epochs:
        loss_value, grad = value_and_grad_fn(
            heuristic_force_params,
            neural_force_params,
            spring_state,
            signed_graph)

        # update neural force params
        neural_force_params = optax.apply_updates(
            neural_force_params,
            optimizer(grad, optimizer_state))

        # update progress bar
        epochs.set_postfix({
            "epoch": epoch_index,
            "loss": loss_value,
        })

    return neural_force_params

def pre_train_loss(
    heuristic_force_params : sim.HeuristicForceParams,
    neural_force_params : nn.NeuralForceParams,
    spring_state : sim.SpringState,
    graph : SignedGraph
) -> jnp.ndarray:
    
    heuristic_force = sim.heuristic_force(
        heuristic_force_params,
        spring_state,
        graph)
    
    neural_force = nn.neural_force(
        neural_force_params,
        spring_state,
        graph)
    
    loss = jnp.mean(jnp.square(heuristic_force - neural_force))

    return loss




