# External dependencies
import sys, getopt
import torch_geometric.transforms as T
from torch_geometric.utils import is_undirected
import yaml
import inquirer
from jax import random, value_and_grad
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.profiler
import optax
import os
import pickle

# Local dependencies
from data import Slashdot, BitcoinO, BitcoinA, WikiRFA, Epinions, Tribes
from graph import permute_split
import simulation as sim
import neural as nn

def main(argv) -> None:
    """
    Main function

    Parameters:
    ----------  
    -s : int (default=64)
        Embedding dimension
    -o : int (default=0)
        Number of iterations for the optimizer
    """
    NN_FORCE = True
    OPTIMIZE_FORCE = True
    EMBEDDING_DIM = 64
    AUXILLARY_DIM = 16
    OPTIMIZE_SPRING_PARAMS = False
    
    NUM_EPOCHS = 400
    NUM_EPOCHS_GRADIENT_ACCUMULATION = 2

    PER_EPOCH_SIM_ITERATIONS = 400
    FINAL_SIM_ITERATIONS = PER_EPOCH_SIM_ITERATIONS
    AUXILLARY_ITERATIONS = 10

    MIN = -10.0
    MAX = 10.0

    DT = 0.01
    DAMPING = 0.3

    DATA_PATH = 'src/data/'
    CECKPOINT_PATH = 'checkpoints/'

    assert not (OPTIMIZE_FORCE and OPTIMIZE_SPRING_PARAMS), "Cannot optimize spring params and use NN force at the same time"
    assert not (not NN_FORCE and OPTIMIZE_FORCE), "Cannot optimize force without using NN force"

    if not OPTIMIZE_FORCE and not OPTIMIZE_SPRING_PARAMS:
        NUM_EPOCHS = 1

    # Deactivate preallocation of memory to avoid OOM errors
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    #os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

    dataset_names = ['Tribes', 'Bitcoin_Alpha', 'BitcoinOTC', 'WikiRFA', 'Slashdot', 'Epinions']
    questions = [
        inquirer.List('dataset',
            message="Choose a dataset",
            choices=dataset_names,
        ),
    ]
    answers = inquirer.prompt(questions)
    dataset_name = answers['dataset']

    opts,i = getopt.getopt(argv,"s:h:d:i:p:o",
        ["embedding_size=","time_step=", "damping=", "iterations="])
    for opt, arg in opts:
        if opt == '-s':
            EMBEDDING_DIM = int(arg)
        elif opt == '-h':
            DT = int(arg)
        elif opt == '-d':
            DAMPING = int(arg)
        elif opt == '-i':
            FINAL_SIM_ITERATIONS = int(arg)

    pre_transform = T.Compose([])
    match dataset_name:
        case "BitcoinOTC":
            dataset = BitcoinO(root=DATA_PATH, pre_transform=pre_transform)
        case "Bitcoin_Alpha":
            dataset = BitcoinA(root=DATA_PATH, pre_transform=pre_transform)
        case "WikiRFA":
            dataset = WikiRFA(root=DATA_PATH, pre_transform=pre_transform)
        case "Slashdot":
            dataset = Slashdot(root=DATA_PATH, pre_transform=pre_transform)
        case "Epinions":
            dataset = Epinions(root=DATA_PATH, pre_transform=pre_transform)
        case "Tribes":
            dataset = Tribes(root=DATA_PATH, pre_transform=pre_transform)

    data = dataset[0]
    if not is_undirected(data.edge_index):
        transform = T.ToUndirected(reduce="min")
        data = transform(data)

    num_nodes = data.num_nodes
    num_edges = data.num_edges

    print(f"num_nodes: {num_nodes}")
    print(f"num_edges: {num_edges}")

    # Permute data and create masks
    # the edges are arranged as follows: training, validation, test
    data, train_mask, val_mask, test_mask = permute_split(data, 0.1, 0.8)

    train_mask = jnp.array(train_mask)
    val_mask = jnp.array(val_mask)
    test_mask = jnp.array(test_mask)
    
    # convert to jnp arrays from torch tensors
    edge_index = jnp.array(data.edge_index)
    signs = jnp.array(data.edge_attr)

    params_path = f"{CECKPOINT_PATH}params_{EMBEDDING_DIM}.yaml"
    print(params_path)
    if os.path.exists(params_path):
        stream = open(params_path, 'r')
        spring_params = yaml.load(stream, Loader=yaml.FullLoader)
        spring_params = sim.SpringParams(**spring_params)
        print("loaded spring params checkpoint")
    else:
        spring_params = sim.SpringParams(
            friend_distance=5.0,
            friend_stiffness=0.3,
            neutral_distance=10.0,
            neutral_stiffness=0.3,
            enemy_distance=20.0,
            enemy_stiffness=0.3,
            distance_threshold=4.0)
        print("no spring params checkpoint found, using default params")

    simulation_params_train = sim.SimulationParams(
        iterations=PER_EPOCH_SIM_ITERATIONS,
        dt=DT,
        damping=DAMPING,
        message_passing_iterations=AUXILLARY_ITERATIONS)

    # Create initial values for neural network parameters
    key_auxillary, key_force, key_training, key_test = random.split(random.PRNGKey(0), 4)

    # auxillary_params = nn.init_attention_params(
    #     key=key_auxillary,
    #     input_dimension=AUXILLARY_DIM + 3,
    #     output_dimension=AUXILLARY_DIM,
    #     factor= 0.01 / AUXILLARY_ITERATIONS)
    
    auxillary_checkpoint_path = f"{CECKPOINT_PATH}auxillary_params_{AUXILLARY_DIM}.yaml"
    if os.path.exists(auxillary_checkpoint_path):
        stream = open(auxillary_checkpoint_path, 'r')
        auxillary_params = yaml.load(stream, Loader=yaml.UnsafeLoader)
        print("loaded auxillary params checkpoint")

    else:
        auxillary_params = nn.init_mlp_params(
            key=key_auxillary,
            layer_dimensions = [AUXILLARY_DIM * 2 + 3,  AUXILLARY_DIM, AUXILLARY_DIM],
            factor= 0.5 / AUXILLARY_ITERATIONS)
        print("no auxillary params checkpoint found, using default params")

    # auxillaries, sign (one hot), difference
    layer_0_size = (AUXILLARY_DIM * 2) + 3
    
    print(f"layer_0_size: {layer_0_size}")
    # params including embdding size
    force_params_name = f"{CECKPOINT_PATH}force_params_{EMBEDDING_DIM}x{AUXILLARY_DIM}.yaml"
    if os.path.exists(force_params_name):
        stream = open(force_params_name, 'r')
        force_params = yaml.load(stream, Loader=yaml.UnsafeLoader)
        print("loaded force params checkpoint")
    else:
        force_params = nn.init_mlp_params(
            key=key_force,
            layer_dimensions = [layer_0_size, layer_0_size, 64, 16, 3],
            factor= 0.5 / PER_EPOCH_SIM_ITERATIONS)
        print("no force params checkpoint found, using default params")
    
    # setup optax optimizers
    if OPTIMIZE_FORCE:
        auxillary_optimizer = optax.adamaxw(learning_rate=1e-4)
        auxillary_multi_step = optax.MultiSteps(auxillary_optimizer, NUM_EPOCHS_GRADIENT_ACCUMULATION)
        auxillary_optimizier_state = auxillary_multi_step.init(auxillary_params)

        force_optimizer = optax.adamaxw(learning_rate=1e-1)
        force_multi_step = optax.MultiSteps(force_optimizer, NUM_EPOCHS_GRADIENT_ACCUMULATION)
        force_optimizier_state = force_multi_step.init(force_params)

        value_grad_fn = value_and_grad(sim.simulate_and_loss, argnums=[4, 5], has_aux=True)

    if OPTIMIZE_SPRING_PARAMS:
        params_optimizer = optax.adamaxw(learning_rate=0.0)
        params_multi_step = optax.MultiSteps(params_optimizer, NUM_EPOCHS_GRADIENT_ACCUMULATION)
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
    if OPTIMIZE_SPRING_PARAMS:
        spring_hist = []    

    epochs = range(NUM_EPOCHS)
    
    epochs_keys = random.split(key_training, NUM_EPOCHS)
    for epoch in epochs:
        # initialize spring state
        # take new key each time to avoid overfitting to specific initial conditions
        spring_state = sim.init_spring_state(
            rng=epochs_keys[epoch],
            min=MIN,
            max=MAX,
            n=data.num_nodes,
            m=data.num_edges,
            embedding_dim=EMBEDDING_DIM,
            auxillary_dim=AUXILLARY_DIM)

        # run simulation and compute loss, auxillaries and gradient
        if OPTIMIZE_FORCE:
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
            
        if OPTIMIZE_SPRING_PARAMS:
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
            

        if OPTIMIZE_FORCE:
            nn_auxillary_update, auxillary_optimizier_state = auxillary_multi_step.update(
                nn_auxillary_grad, auxillary_optimizier_state, auxillary_params)
        
            auxillary_params = optax.apply_updates(auxillary_params, nn_auxillary_update)
        
            nn_force_update, force_optimizier_state = force_multi_step.update(
                nn_force_grad, force_optimizier_state, force_params)
            
            force_params = optax.apply_updates(force_params, nn_force_update)

        if OPTIMIZE_SPRING_PARAMS:
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

            if OPTIMIZE_SPRING_PARAMS:
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
    if OPTIMIZE_SPRING_PARAMS:
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

    # write spring params to file, the file is still a traced jax object
    if OPTIMIZE_SPRING_PARAMS:
        # ask user if they want to save the parameters
        questions = [
            inquirer.List('save',
                message="Do you want to save the spring parameters?",
                choices=['Yes', 'No'],
            ),
        ]
        answers = inquirer.prompt(questions)
        if answers['save'] == 'Yes':
            with open("src/params.yaml", 'w') as file:
                params_dict = {}
                # get value from jax traced array
                params_dict['friend_distance'] = spring_params.friend_distance.item()
                params_dict['friend_stiffness'] = spring_params.friend_stiffness.item()
                params_dict['neutral_distance'] = spring_params.neutral_distance.item()
                params_dict['neutral_stiffness'] = spring_params.neutral_stiffness.item()
                params_dict['enemy_distance'] = spring_params.enemy_distance.item()
                params_dict['enemy_stiffness'] = spring_params.enemy_stiffness.item()
                params_dict['distance_threshold'] = spring_params.distance_threshold.item()

                print(params_dict)

                yaml.dump(params_dict, file)

    # Store the trained parameters in a file
    if OPTIMIZE_FORCE:
        # ask user if they want to save the parameters
        questions = [
            inquirer.List('save',
                message="Do you want to save the neural network parameters?",
                choices=['Yes', 'No'],
            ),
        ]
        answers = inquirer.prompt(questions)
        if answers['save'] == 'Yes':
                
            with open(auxillary_checkpoint_path, 'w') as file:
                yaml.dump(auxillary_params, file)

            with open(force_params_name, 'w') as file:
                yaml.dump(force_params, file)

    spring_state = sim.init_spring_state(
        rng=key_test,
        n=data.num_nodes,
        m=data.num_edges,
        min=MIN,
        max=MAX,
        embedding_dim=EMBEDDING_DIM,
        auxillary_dim=AUXILLARY_DIM
    )

    training_signs = signs.copy()
    training_signs = training_signs.at[train_mask].set(0)

    simulation_params_test = sim.SimulationParams(
        iterations=FINAL_SIM_ITERATIONS,
        dt=DT,
        damping=DAMPING,
        message_passing_iterations=1)

    spring_state = sim.simulate(
        simulation_params=simulation_params_test,
        spring_state=spring_state, 
        spring_params=spring_params,
        nn_force=NN_FORCE,
        nn_auxillary_params=auxillary_params,
        nn_force_params=force_params,
        edge_index=edge_index,
        sign=training_signs)

    # metrics = sim.evaluate(
    #     spring_state,
    #     edge_index,
    #     signs,
    #     train_mask,
    #     test_mask)

    # print(metrics)

    # # create four subplots
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # embeddings = spring_state.position
    # # plot the embeddings
    # ax1.scatter(embeddings[:, 0], embeddings[:, 1])# c=spring_state.energy)
    # # color bar
    # sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    # sm.set_array([])
    # fig.colorbar(sm, ax=ax1)
    # ax1.set_title('Embeddings')

    # # # plot energies
    # # ax2.hist(spring_state.energy)
    # # ax2.set_title('Energies')

    # # plot the energies over time, log scale
    # ax3.plot(total_energies)
    # ax3.set_yscale('log')
    # ax3.set_title('Total energy')

    # # plot measures
    # ax4.plot(aucs)
    # ax4.plot(f1_binaries)
    # ax4.plot(f1_micros)
    # ax4.plot(f1_macros)
    # ax4.set_title('Measures')
    # ax4.legend(['AUC', 'F1 binary', 'F1 micro', 'F1 macro'])

    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])
    