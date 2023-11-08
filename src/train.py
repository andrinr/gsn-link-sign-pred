import jax.numpy as jnp
import optax
from jax import random, value_and_grad
from typing import NamedTuple
import simulation as sim

class TrainParams(NamedTuple):
    use_nn_force : bool
    optimize_force : bool
    optimize_params : bool
    num_epochs : int
    num_epochs_gradient_accumulation : int
    embedding_dim : int
    auxillary_dim : int
    num_message_passing_iterations : int
    position_init_range : float

def train(
    edge_index : jnp.ndarray,
    sign : jnp.ndarray,
    train_mask : jnp.ndarray,
    test_mask : jnp.ndarray,
    val_mask : jnp.ndarray,
    params : TrainParams):

    num_edges = edge_index.shape[1]
    num_nodes = 

    # setup optax optimizers
    if params.optimize_force:
        auxillary_optimizer = optax.adamaxw(learning_rate=1e-5)
        auxillary_multi_step = optax.MultiSteps(auxillary_optimizer, params.num_epochs_gradient_accumulation)
        auxillary_optimizier_state = auxillary_multi_step.init(auxillary_params)

        force_optimizer = optax.adamaxw(learning_rate=1e-4)
        force_multi_step = optax.MultiSteps(force_optimizer, params.num_epochs_gradient_accumulation)
        force_optimizier_state = force_multi_step.init(force_params)

        value_grad_fn = value_and_grad(sim.simulate_and_loss, argnums=[4, 5], has_aux=True)

    if params.optimize_params:
        params_optimizer = optax.adamaxw(learning_rate=0.1)
        params_multi_step = optax.MultiSteps(params_optimizer, params.num_epochs_gradient_accumulation)
        params_optimizier_state = params_multi_step.init(spring_params)  

        value_grad_fn = value_and_grad(sim.simulate_and_loss, argnums=2, has_aux=True)

    total_energies = []
    aucs = []
    f1_binaries = []
    f1_micros = []
    f1_macros = []

    loss_hist = []
    metrics_hist = []
    loss_mov_avg = 0.0
    if params.optimize_params:
        spring_hist = []    

    epochs = range(params.num_epochs)
    
    epochs_keys = random.split(key_training, params.num_epochs)
    for epoch in epochs:
        # initialize spring state
        # take new key each time to avoid overfitting to specific initial conditions
        spring_state = sim.init_spring_state(
            rng=epochs_keys[epoch],
            min=-params.position_init_range,
            max=params.position_init_range,
            n=data.num_nodes,
            m=data.num_edges,
            embedding_dim=EMBEDDING_DIM,
            auxillary_dim=AUXILLARY_DIM)

        # run simulation and compute loss, auxillaries and gradient
        if optimize_force:
            (loss_value, (spring_state, signs_pred)), (nn_auxillary_grad, nn_force_grad) = value_grad_fn(
                simulation_params_train, #0
                spring_state, #1
                spring_params, #2
                NN_FORCE, #3
                auxillary_params, #4
                force_params, #5
                edge_index, #6
                signs, #7
                train_mask, #8
                val_mask)
            
        if optimize_params:
            (loss_value, (spring_state, signs_pred)), params_grad = value_grad_fn(
                simulation_params_train, #0
                spring_state, #1
                spring_params, #2
                NN_FORCE, #3
                auxillary_params, #4
                force_params, #5
                edge_index, #6
                signs, #7
                train_mask, #8
                val_mask)
        else:
            loss_value, (spring_state, signs_pred) = sim.simulate_and_loss(
                simulation_params_train, #0
                spring_state, #1
                spring_params, #2
                NN_FORCE, #3
                auxillary_params, #4
                force_params, #5
                edge_index, #6
                signs, #7
                train_mask, #8
                val_mask)
            

        if optimize_force:
            nn_auxillary_update, auxillary_optimizier_state = auxillary_multi_step.update(
                nn_auxillary_grad, auxillary_optimizier_state, auxillary_params)
        
            auxillary_params = optax.apply_updates(auxillary_params, nn_auxillary_update)
        
            nn_force_update, force_optimizier_state = force_multi_step.update(
                nn_force_grad, force_optimizier_state, force_params)
            
            force_params = optax.apply_updates(force_params, nn_force_update)

        if optimize_params:
            params_update, params_optimizier_state = params_multi_step.update(
                params_grad, params_optimizier_state, spring_params)
            
            spring_params = optax.apply_updates(spring_params, params_update)

        signs_ = signs * 0.5 + 0.5
        metrics = sim.evaluate(
            spring_state,
            edge_index,
            signs,
            train_mask,
            val_mask)
        
        loss_mov_avg += loss_value

        if epoch % NUM_EPOCHS_GRADIENT_ACCUMULATION == 0:
            loss_mov_avg = loss_mov_avg / NUM_EPOCHS_GRADIENT_ACCUMULATION
            print(metrics)
            print(f"epoch: {epoch}")
            print(f"predictions: {signs_pred}")
            print(f"loss: {loss_value}")
            print(f"loss_mov_avg: {loss_mov_avg}")
            print(f"correct predictions: {jnp.sum(jnp.equal(jnp.round(signs_pred), signs_))} out of {signs.shape[0]}")

            loss_hist.append(loss_mov_avg)
            metrics_hist.append(metrics)

            loss_mov_avg = 0.0

            if optimize_params:
                spring_hist.append(spring_params)

    # # plot the embeddings
    # plt.scatter(spring_state.position[:, 0], spring_state.position[:, 1])
    # # add edges to plot
    # for i in range(edge_index.shape[1]):
    #     plt.plot(
    #         [spring_state.position[edge_index[0, i], 0], spring_state.position[edge_index[1, i], 0]],
    #         [spring_state.position[edge_index[0, i], 1], spring_state.position[edge_index[1, i], 1]],
    #         color= 'blue' if signs[i] == 1 else 'red',
    #         alpha=0.5)
        
    # # add legend for edges
    # plt.plot([], [], color='blue', label='positive')
    # plt.plot([], [], color='red', label='negative')
    # plt.legend()

    # plt.show()

    # plot loss over time
    spaced_epochs = range(0, NUM_EPOCHS, NUM_EPOCHS_GRADIENT_ACCUMULATION)
    plt.plot(spaced_epochs, loss_hist)
    plt.title('Loss')
    plt.show()

    # plot metrics over time
    plt.plot(spaced_epochs, [metrics.auc for metrics in metrics_hist])
    plt.plot(spaced_epochs, [metrics.f1_binary for metrics in metrics_hist])
    plt.plot(spaced_epochs, [metrics.f1_micro for metrics in metrics_hist])
    plt.plot(spaced_epochs, [metrics.f1_macro for metrics in metrics_hist])
    plt.legend(['AUC', 'F1 binary', 'F1 micro', 'F1 macro'])
    plt.title('Measures')
    plt.show()

    # plot spring params over time
    if optimize_params:
        plt.plot(spaced_epochs, [spring_params.friend_distance for spring_params in spring_hist])
        plt.plot(spaced_epochs, [spring_params.friend_stiffness for spring_params in spring_hist])
        plt.plot(spaced_epochs, [spring_params.neutral_distance for spring_params in spring_hist])
        plt.plot(spaced_epochs, [spring_params.neutral_stiffness for spring_params in spring_hist])
        plt.plot(spaced_epochs, [spring_params.enemy_distance for spring_params in spring_hist])
        plt.plot(spaced_epochs, [spring_params.enemy_stiffness for spring_params in spring_hist])
        plt.plot(spaced_epochs, [spring_params.distance_threshold for spring_params in spring_hist])
        plt.legend(['friend_distance', 'friend_stiffness', 'neutral_distance', 'neutral_stiffness', 'enemy_distance', 'enemy_stiffness', 'distance_threshold'])
        plt.title('Spring params')
        plt.show()