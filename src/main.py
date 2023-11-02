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
    NN_FORCE = True
    EMBEDDING_DIM = 16
    NN_AUXILLARY = True
    AUXILLARY_DIM = 16
    OPTIMIZE_SPRING_PARAMS = False

    DT = 0.02
    DAMPING = 0.1

    NUM_EPOCHS = 5000
    PER_EPOCH_SIM_ITERATIONS = 200
    FINAL_SIM_ITERATIONS = 200
    AUXILLARY_ITERATIONS = 6

    DATA_ROOT = 'src/data/'

    # Deactivate preallocation of memory to avoid OOM errors
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    #os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

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
            dataset = BitcoinO(root=DATA_ROOT, pre_transform=pre_transform)
        case "Bitcoin_Alpha":
            dataset = BitcoinA(root=DATA_ROOT, pre_transform=pre_transform)
        case "WikiRFA":
            dataset = WikiRFA(root=DATA_ROOT, pre_transform=pre_transform)
        case "Slashdot":
            dataset = Slashdot(root=DATA_ROOT, pre_transform=pre_transform)
        case "Epinions":
            dataset = Epinions(root=DATA_ROOT, pre_transform=pre_transform)
        case "Tribes":
            dataset = Tribes(root=DATA_ROOT, pre_transform=pre_transform)

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

    stream = open("src/params.yaml", 'r')
    params = yaml.load(stream, Loader=yaml.FullLoader)
    spring_params = sim.SpringParams(
        friend_distance=1.0,
        friend_stiffness=4.0,
        enemy_distance=21.0,
        enemy_stiffness=5.0)
    
    simulation_params_train = sim.SimulationParams(
        iterations=PER_EPOCH_SIM_ITERATIONS,
        dt=DT,
        damping=DAMPING,
        message_passing_iterations=AUXILLARY_ITERATIONS)

    # Create initial values for neural network parameters
    key_attention, key_mlp, key_training, key_test = random.split(random.PRNGKey(0), 4)

    auxillary_params = nn.init_attention_params(
        key=key_attention,
        input_dimension=AUXILLARY_DIM + 3,
        output_dimension=AUXILLARY_DIM,
        factor=1 / AUXILLARY_ITERATIONS)

    layer_0_size = (int(NN_AUXILLARY) * AUXILLARY_DIM * 2) + 4
    print(f"layer_0_size: {layer_0_size}")
    force_params = nn.init_mlp_params(
        key=key_mlp,
        layer_dimensions = [layer_0_size, layer_0_size, 1],
        factor=1 / PER_EPOCH_SIM_ITERATIONS)
    
    # setup optax optimizers
    if NN_AUXILLARY:
        auxillary_optimizer = optax.radam(learning_rate=1e-4)
        auxillary_optimizier_state = auxillary_optimizer.init(auxillary_params)

    if NN_FORCE:
        force_optimizer = optax.radam(learning_rate=1e-4)
        force_optimizier_state = force_optimizer.init(force_params)

    # compute value and grad function of simulation using jax
    argnums = []
    if NN_AUXILLARY: argnums.append(4)
    if NN_FORCE: argnums.append(6)
    if len(argnums) == 1: argnums = argnums[0]

    if NN_AUXILLARY or NN_FORCE:
        value_grad_fn = value_and_grad(sim.simulate_and_loss, argnums=argnums, has_aux=True)

    total_energies = []
    aucs = []
    f1_binaries = []
    f1_micros = []
    f1_macros = []
    
    epochs_keys = random.split(key_training, NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        # initialize spring state
        # take new key each time to avoid overfitting to specific initial conditions
        spring_state = sim.init_spring_state(
            rng=epochs_keys[epoch],
            n=data.num_nodes,
            m=data.num_edges,
            embedding_dim=EMBEDDING_DIM,
            auxillary_dim=AUXILLARY_DIM)

        # run simulation and compute loss, auxillaries and gradient
        if NN_AUXILLARY or NN_FORCE:
            (loss_value, (spring_state, signs_pred)), grads = value_grad_fn(
                simulation_params_train, #0
                spring_state, #1
                spring_params, #2
                NN_AUXILLARY, #3
                auxillary_params, #4
                NN_FORCE, #5
                force_params, #6
                edge_index, #7
                signs, #8
                train_mask, #9
                val_mask)
        else:
            loss_value, (spring_state, signs_pred) = sim.simulate_and_loss(
                simulation_params_train, #0
                spring_state, #1
                spring_params, #2
                NN_AUXILLARY, #3
                auxillary_params, #4
                NN_FORCE, #5
                force_params, #6
                edge_index, #7
                signs, #8
                train_mask, #9
                val_mask)
        
        if NN_AUXILLARY and NN_FORCE:
            (nn_auxillary_grad, nn_force_grad) = grads

        elif NN_AUXILLARY:
            nn_auxillary_grad = grads

        elif NN_FORCE:
            nn_force_grad = grads

        # make sure there are no zero gradients 
        # print(f"signs_pred: {signs_pred}")
        # print(f"signs: {signs}")
        # # print(f"springs: {spring_state}")
        # # print(f"auxillaries_params: {auxillaries_params}")
        # # print(f"forces_params: {forces_params}")

        if NN_AUXILLARY:
            nn_auxillary_update, auxillary_optimizier_state = auxillary_optimizer.update(
                nn_auxillary_grad, auxillary_optimizier_state, auxillary_params)
        
            auxillary_params = optax.apply_updates(auxillary_params, nn_auxillary_update)
        
        if NN_FORCE:
            nn_force_update, force_optimizier_state = force_optimizer.update(
                nn_force_grad, force_optimizier_state, force_params)
            
            force_params = optax.apply_updates(force_params, nn_force_update)

        # print(f"auxillaries_params: {auxillaries_params}")
        # print(f"forces_params: {forces_params}")

        # if epoch % 30 == 0:
        signs_ = signs * 0.5 + 0.5
        print(f"predictions: {signs_pred}")
        print(f"loss: {loss_value}")
        print(f"correct predictions: {jnp.sum(jnp.equal(jnp.round(signs_pred), signs_))} out of {signs.shape[0]}")

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

        # metrics = sim.evaluate(
        #     spring_state,
        #     edge_index,
        #     signs,
        #     train_mask,
        #     val_mask)
        
        # print(metrics)
        
        # aucs.append(metrics.auc)
        # f1_binaries.append(metrics.f1_binary)
        # f1_micros.append(metrics.f1_micro)
        # f1_macros.append(metrics.f1_macro)

    jax.profiler.save_device_memory_profile("memory.prof")
    
    spring_state = sim.init_spring_state(
        rng=key_test,
        n=data.num_nodes,
        m=data.num_edges,
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
        nn_auxillary=NN_AUXILLARY,
        nn_auxillary_params=auxillary_params,
        nn_force=NN_FORCE,
        nn_force_params=force_params,
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
    