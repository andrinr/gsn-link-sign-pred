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
    embedding_dim = 8
    num_trainings = 2
    training_simulation_iterations = 20
    simulation_iterations = 200
    time_step =  0.1
    damping = 0.1
    root = 'src/data/'

    # Deactivate preallocation of memory to avoid OOM errors
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    #os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
    #os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

    dataset_names = ['Bitcoin_Alpha', 'BitcoinOTC', 'WikiRFA', 'Slashdot', 'Epinions', 'Tribes']
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
            embedding_dim = int(arg)
        elif opt == '-h':
            time_step = int(arg)
        elif opt == '-d':
            damping = int(arg)
        elif opt == '-i':
            simulation_iterations = int(arg)

    stream = open("src/params.yaml", 'r')
    params = yaml.load(stream, Loader=yaml.FullLoader)

    pre_transform = T.Compose([])
    match dataset_name:
        case "BitcoinOTC":
            dataset = BitcoinO(root= root, pre_transform=pre_transform)
        case "Bitcoin_Alpha":
            dataset = BitcoinA(root= root, pre_transform=pre_transform)
        case "WikiRFA":
            dataset = WikiRFA(root= root, pre_transform=pre_transform)
        case "Slashdot":
            dataset = Slashdot(root= root, pre_transform=pre_transform)
        case "Epinions":
            dataset = Epinions(root= root, pre_transform=pre_transform)
        case "Tribes":
            dataset = Tribes(root= root, pre_transform=pre_transform)

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

    spring_params = sim.SpringParams(
        friend_distance=5.0,
        friend_stiffness=5.0,
        enemy_distance=20.0,
        enemy_stiffness=6.0)
    
    simulation_params_train = sim.SimulationParams(
        iterations=training_simulation_iterations,
        dt=time_step,
        damping=damping,
        message_passing_iterations=1)

    # Create initial values for neural network parameters
    key_attention, key_mlp, key_training, key_test = random.split(random.PRNGKey(0), 4)

    auxillaries_params = nn.init_attention_params(
        key=key_attention,
        input_dimension=embedding_dim + 1,
        output_dimension=embedding_dim)
    
    forces_params = nn.init_mlp_params(
        key=key_mlp,
        layer_dimensions = [embedding_dim * 3 + 1, 128, 32, 1])
    
    # setup optax optimizers
    auxillaries_opt = optax.adam(learning_rate=1e-2)
    forces_op = optax.adam(learning_rate=1e-2)

    auxillaries_opt_state = auxillaries_opt.init(auxillaries_params)
    forces_opt_state = forces_op.init(forces_params)

    # compute value and grad function of simulation using jax
    value_grad_fn = value_and_grad(sim.simulate_and_loss, argnums=[4, 5], has_aux=True)
    total_energies = []
    aucs = []
    f1_binaries = []
    f1_micros = []
    f1_macros = []  
    
    keys_training = random.split(key_training, num_trainings)
    for i in range(num_trainings):
        # initialize spring state
        # take new key each time to avoid overfitting to specific initial conditions
        spring_state = sim.init_spring_state(
            rng=keys_training[i],
            n=data.num_nodes,
            m=data.num_edges,
            embedding_dim=embedding_dim,
        )

        # run simulation and compute loss, auxillaries and gradient
        (loss_value, spring_state), (auxillaries_gradient, forces_gradient) = value_grad_fn(
            simulation_params_train,
            spring_state, 
            spring_params,
            False, # nn_based_forces
            auxillaries_params,
            forces_params,
            edge_index,
            signs,
            train_mask,
            val_mask)
        
        # print(f"springs: {spring_state}")
        # print(f"auxillaries_params: {auxillaries_params}")
        # print(f"forces_params: {forces_params}")
        
        # print(f"auxillaries_gradient: {auxillaries_gradient}")
        # print(f"forces_gradient: {forces_gradient}")
        
        auxillaries_updates, auxillaries_opt_state = auxillaries_opt.update(
            auxillaries_gradient, auxillaries_opt_state, auxillaries_params)
        
        auxillaries_params = optax.apply_updates(auxillaries_params, auxillaries_updates)

        forces_updates, forces_opt_state = forces_op.update(
            forces_gradient, forces_opt_state, forces_params)
        
        forces_params = optax.apply_updates(forces_params, forces_updates)

        # print(f"auxillaries_params: {auxillaries_params}")
        # print(f"forces_params: {forces_params}")
        
        # print(f"loss: {loss_value}")

        # # update spring params
        # spring_params = spring_params._replace(
        #     friend_distance=spring_params.friend_distance - learning_rate * parameter_gradient.friend_distance,
        #     friend_stiffness=spring_params.friend_stiffness - learning_rate * parameter_gradient.friend_stiffness,
        #     enemy_distance=spring_params.enemy_distance - learning_rate * parameter_gradient.enemy_distance,
        #     enemy_stiffness=spring_params.enemy_stiffness - learning_rate * parameter_gradient.enemy_stiffness
        # )

        # print(f"friend_distance gradient: {parameter_gradient.friend_distance}")
        # print(f"friend_stiffness gradient: {parameter_gradient.friend_stiffness}")
        # print(f"enemy_distance gradient: {parameter_gradient.enemy_distance}")
        # print(f"enemy_stiffness gradient: {parameter_gradient.enemy_stiffness}")

        # print(f"friend_distance: {spring_params.friend_distance}")
        # print(f"friend_stiffness: {spring_params.friend_stiffness}")
        # print(f"enemy_distance: {spring_params.enemy_distance}")
        # print(f"enemy_stiffness: {spring_params.enemy_stiffness}")

        metrics = sim.evaluate(
            spring_state,
            edge_index,
            signs,
            train_mask,
            val_mask)
        
        print(metrics)
        
        aucs.append(metrics.auc)
        f1_binaries.append(metrics.f1_binary)
        f1_micros.append(metrics.f1_micro)
        f1_macros.append(metrics.f1_macro)

    jax.profiler.save_device_memory_profile("memory.prof")
    
    spring_state = sim.init_spring_state(
        rng=key_test,
        n=data.num_nodes,
        m=data.num_edges,
        embedding_dim=embedding_dim,
    )

    training_signs = signs.copy()
    training_signs = training_signs.at[train_mask].set(0)

    simulation_params_test = sim.SimulationParams(
        iterations=simulation_iterations,
        dt=time_step,
        damping=damping,
        message_passing_iterations=1)

    spring_state = sim.simulate(
        simulation_params=simulation_params_test,
        spring_state=spring_state, 
        spring_params=spring_params,
        nn_based_forces=True,
        auxillaries_nn_params=auxillaries_params,
        forces_nn_params=forces_params,
        edge_index=edge_index,
        signs=training_signs)

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
    