from typing import NamedTuple
import jax
import jax.numpy as jnp
from tqdm import tqdm
import optax
from jax import custom_vjp
import nevergrad as ng

import graph as g
import simulation as sm

class TrainingParams(NamedTuple):
    num_epochs : int
    learning_rate : float
    batch_size : int
    init_pos_range : float
    embedding_dim : int
    multi_step : int
    blackbox : bool

@custom_vjp
def clip_gradient(lo, hi, x):
  return x  # identity function

def clip_gradient_fwd(lo, hi, x):
  return x, (lo, hi)  # save bounds as residuals

def clip_gradient_bwd(res, g):
  lo, hi = res
  return (None, None, jnp.clip(g, lo, hi))  # use None to indicate zero cotangents for lo and hi

clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)

def train(
    random_key : jax.random.PRNGKey,
    batches : list[g.SignedGraph],
    use_neural_force : bool,
    force_params : sm.NeuralEdgeParams | sm.SpringForceParams,
    training_params : TrainingParams,
    simulation_params : sm.SimulationParams,
    ) -> tuple[sm.NeuralEdgeParams, list[float], list[sm.Metrics]]:
   
    if training_params.blackbox:
        return blackbox_training(
            random_key,
            batches,
            use_neural_force,
            force_params,
            training_params,
            simulation_params)
    else:
        return gradient_training(
            random_key,
            batches,
            use_neural_force,
            force_params,
            training_params,
            simulation_params)
    
def blackbox_training(
    random_key : jax.random.PRNGKey,
    batches : list[g.SignedGraph],
    use_neural_force : bool,
    force_params : sm.NeuralEdgeParams | sm.SpringForceParams,
    training_params : TrainingParams,
    simulation_params : sm.SimulationParams,
) -> tuple[sm.NeuralEdgeParams, list[float], list[sm.Metrics]]:
   
    optimizer = ng.optimizers.NGOpt(parametrization=2, budget=training_params.num_epochs)

    recomd = optimizer.minimize(
        lambda x: sm.simulate_and_loss(
            simulation_params,
            sm.init_node_state(
                rng=random_key,
                range=training_params.init_pos_range,
                n=batches[0].num_nodes,
                m=batches[0].num_edges,
                embedding_dim=training_params.embedding_dim),
            use_neural_force,
            x,
            batches[0])[0])
    
    print(recomd)

       
def gradient_training(
    random_key : jax.random.PRNGKey,
    batches : list[g.SignedGraph],
    use_neural_force : bool,
    force_params : sm.NeuralEdgeParams | sm.SpringForceParams,
    training_params : TrainingParams,
    simulation_params : sm.SimulationParams,
) -> tuple[sm.NeuralEdgeParams, list[float], list[sm.Metrics]]:

    optimizer = optax.adam(training_params.learning_rate)
    # optimizer = optax.noisy_sgd(
    #     learning_rate=training_params.learning_rate,
    #     eta=0.01,
    #     gamma=0.55)
    
    force_optimizer_multi_step = optax.MultiSteps(
        optimizer, training_params.multi_step)
    force_optimizier_state = force_optimizer_multi_step.init(
        force_params)
    
    value_and_grad_fn = jax.value_and_grad(sm.simulate_and_loss, argnums=3, has_aux=True)
    
    epochs = tqdm(range(training_params.num_epochs))

    loss_history = []
    metrics_history = []
    force_params_history = []
    epoch_loss = 0

    random_keys = jax.random.split(random_key, training_params.num_epochs)

    for epoch_index in epochs:

        for batch_index, batch_graph in enumerate(batches):
            # initialize spring state
            # take new key each time to avoid overfitting to specific initial condition
            spring_state = sm.init_node_state(
                rng=random_keys[0],
                range=training_params.init_pos_range,
                n=batch_graph.num_nodes,
                m=batch_graph.num_edges,
                embedding_dim=training_params.embedding_dim)
            # run simulation and compute loss, auxillaries and gradient
            (loss_value, (spring_state, signs_pred)), grad = value_and_grad_fn(
                simulation_params, #0
                spring_state, #1
                use_neural_force, #2
                force_params, #3
                batch_graph)
            
            grad = clip_gradient(-1, 1, grad)
            
            nn_force_update, force_optimizier_state = force_optimizer_multi_step.update(
                grad, force_optimizier_state, force_params)
            
            force_params = optax.apply_updates(force_params, nn_force_update)

            epoch_loss += loss_value

            if epoch_index % 10 == 0:
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
                # print 5 digits after comma
                "loss": f"{loss_value:.5f}",
                "auc_p": round(metrics.auc_p, 5),
                "auc_l": round(metrics.auc_l, 5),
                "f1_macro": round(metrics.f1_macro, 5),
            })

        loss_history.append(epoch_loss)
        epoch_loss = 0
        metrics_history.append(metrics)
        force_params_history.append(force_params)

    return force_params, loss_history, metrics_history, force_params_history
