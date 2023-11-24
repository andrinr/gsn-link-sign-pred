import jax
import jax.numpy as jnp
from tqdm import tqdm
import optax

# local imports
import graph as g
import simulation as sm

def pre_train(
    key : jax.random.PRNGKey,
    learning_rate : float,
    num_epochs : int,
    heuristic_force_params : sm.NeuralForceParams,
    neural_force_params : sm.NeuralForceParams,
) -> sm.NeuralForceParams:
    
    print("Pre-training neural force... \n")

    # generate random SignedGraph
    num_edges = 100000

    sign = jax.random.randint(
        key, 
        minval=-1, 
        maxval=2, 
        shape=(num_edges,))
    
    sign_one_hot = jax.nn.one_hot(sign + 1, 3)
    
    distances = jax.random.uniform(
        key,
        minval=-50,
        maxval=50,
        shape=(num_edges,1))
    
    true_force = sm.heuristic_force(
        heuristic_force_params,
        distances,
        sign)

    # create optimizer
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(neural_force_params)

    # create tqdm progress bar
    epochs = tqdm(range(num_epochs))

    # create value and grad function
    value_and_grad_fn = jax.value_and_grad(pre_train_loss, argnums=0, has_aux=True)

    # pre train
    for epoch_index in epochs:
        (loss_value, neural_force), grad = value_and_grad_fn(
            neural_force_params,
            true_force,
            sign_one_hot,
            distances)

        updates, optimizer_state = optimizer.update(grad, optimizer_state, neural_force_params)

        # print(loss_value)
        # print(neural_force)
        # print(true_force)
        # update neural force params
        neural_force_params = optax.apply_updates(
            neural_force_params,
            updates)

        # update progress bar
        epochs.set_postfix({
            "epoch": epoch_index,
            "loss": loss_value,
        })

    return neural_force_params

def pre_train_loss(
    neural_force_params : sm.NeuralForceParams,
    true_force : jnp.ndarray,
    sign_one_hot : jnp.ndarray,
    distance : jnp.ndarray,
) -> jnp.ndarray:
    
    neural_force = sm.neural_force(
        neural_force_params,
        distance,
        sign_one_hot)

    loss = jnp.mean(jnp.square(true_force - neural_force))

    return loss, neural_force