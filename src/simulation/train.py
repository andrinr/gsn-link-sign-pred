from typing import NamedTuple
import jax
import jax.numpy as jnp
from tqdm import tqdm
import optax

import graph as g
import simulation as sm

class TrainingParams(NamedTuple):
    num_epochs : int
    learning_rate : float
    batch_size : int
    init_pos_range : float
    embedding_dim : int
    multi_step : int

def train(
    random_key : jax.random.PRNGKey,
    batches : list[g.SignedGraph],
    force_params : sm.HeuristicForceParams,
    training_params : TrainingParams,
    simulation_params : sm.SimulationParams,
) -> (sm.HeuristicForceParams, list[float], list[sm.Metrics]):

    print("\n Training...")

    optimizer = optax.adam(training_params.learning_rate)
    force_optimizer_multi_step = optax.MultiSteps(
        optimizer, training_params.multi_step)
    force_optimizier_state = force_optimizer_multi_step.init(
        force_params)
    
    value_and_grad_fn = jax.value_and_grad(sm.simulate_and_loss, argnums=2, has_aux=True)
    
    # init progress bar, add total number of epochs, add loss and metrics
    epochs = tqdm(range(training_params.num_epochs))

    loss_history = []
    metrics_history = []
    force_params_history = []
    epoch_loss = 0

    random_keys = jax.random.split(random_key, training_params.num_epochs)

    for epoch_index in epochs:
        for batch_index, batch_graph in enumerate(batches):
            # initialize spring state
            # take new key each time to avoid overfitting to specific initial conditions
            spring_state = sm.init_spring_state(
                rng=random_keys[0],
                range=training_params.init_pos_range,
                n=batch_graph.num_nodes,
                m=batch_graph.num_edges,
                embedding_dim=training_params.embedding_dim)

            # run simulation and compute loss, auxillaries and gradient
            (loss_value, (spring_state, signs_pred)), grad = value_and_grad_fn(
                simulation_params, #0
                spring_state, #1
                force_params, #2
                batch_graph)
        
            print(force_params)
            print(grad)
            print(loss_value)

            nn_force_update, force_optimizier_state = force_optimizer_multi_step.update(
                grad, force_optimizier_state, force_params)
            
            force_params = optax.apply_updates(force_params, nn_force_update)

            print(force_params)
            epoch_loss += loss_value

            metrics, _= sm.evaluate(
                spring_state,
                batch_graph.edge_index,
                batch_graph.sign,
                batch_graph.train_mask,
                batch_graph.test_mask)

            # update progress bar
            epochs.set_postfix({
                "epoch": epoch_index,
                "batch": batch_index,
                "loss": loss_value,
                "auc": metrics.auc,
                "f1_binary": metrics.f1_binary,
                "f1_micro": metrics.f1_micro,
                "f1_macro": metrics.f1_macro,
            })

        loss_history.append(epoch_loss)
        epoch_loss = 0
        metrics_history.append(metrics)
        force_params_history.append(force_params)

    return force_params, loss_history, metrics_history, force_params_history
