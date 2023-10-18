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

# Local dependencies
from data import Slashdot, BitcoinO, BitcoinA, WikiRFA, Epinions
from graph import permute_split
from . import simulate
from springs import SpringParams, evaluate, init_spring_state, simulate_and_loss, SimulationParams
from gnn import AttentionHead

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
    embedding_dim = 64
    trainings = 50
    training_iterations = 100
    iterations = 200
    time_step =  0.01
    damping = 0.1
    n_attention_heads = 3
    root = 'src/data/'

    dataset_names = ['Bitcoin_Alpha', 'BitcoinOTC', 'WikiRFA', 'Slashdot', 'Epinions']
    questions = [
        inquirer.List('dataset',
            message="Choose a dataset",
            choices=dataset_names,
        ),
    ]
    answers = inquirer.prompt(questions)
    dataset_name = answers['dataset']

    opts,_ = getopt.getopt(argv,"s:h:d:i:p:o",
        ["embedding_size=","time_step=", "damping=", "iterations="])
    for opt, arg in opts:
        if opt == '-s':
            embedding_dim = int(arg)
        elif opt == '-h':
            time_step = int(arg)
        elif opt == '-d':
            damping = int(arg)
        elif opt == '-i':
            iterations = int(arg)

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

    data = dataset[0]
    if not is_undirected(data.edge_index):
        transform = T.ToUndirected(reduce="min")
        data = transform(data)

    # Permute data and create masks
    # the edges are arranged as follows: training, validation, test
    data, train_mask, val_mask, test_mask = permute_split(data, 0.1, 0.8)

    train_mask = jnp.array(train_mask)
    val_mask = jnp.array(val_mask)
    test_mask = jnp.array(test_mask)
    
    # convert to jnp arrays from torch tensors
    edge_index = jnp.array(data.edge_index)
    signs = jnp.array(data.edge_attr)

    spring_params = SpringParams(
        friend_distance=5.0,
        friend_stiffness=5.0,
        enemy_distance=20.0,
        enemy_stiffness=6.0,
    )

    simulation_params = SimulationParams(
        iterations=training_iterations,
        dt=time_step,
        damping=damping,
        message_passing_iterations=5
    )

    attention_head = AttentionHead(embedding_dimensions=embedding_dim)
    attention_head_params = []
    rng = random.PRNGKey(42)
    keys = random.split(rng, n_attention_heads)
    for i in range(n_attention_heads):
        attention_head_params.append(attention_head.init(keys[i]))
    
    total_energies = []
    aucs = []
    f1_binaries = []
    f1_micros = []
    f1_macros = []
    
    grad = value_and_grad(simulate_and_loss, argnums=2, has_aux=True)
    
    learning_rate = 1.0
    for i in range(trainings):
        spring_state = init_spring_state(
            rng=rng,
            n=data.num_nodes,
            m=data.num_edges,
            embedding_dim=embedding_dim,
        )

        (loss_value, spring_state), parameter_gradient = grad(
            simulation_params,
            spring_state, 
            spring_params,
            attention_head,
            attention_head_params,
            edge_index,
            signs,
            train_mask,
            val_mask,)
        
        print(f"loss: {loss_value}")
        print(f"parameter_gradient: {parameter_gradient}")

        # update spring state
        spring_params = spring_params._replace(
            friend_distance=spring_params.friend_distance - learning_rate * parameter_gradient.friend_distance,
            friend_stiffness=spring_params.friend_stiffness - learning_rate * parameter_gradient.friend_stiffness,
            enemy_distance=spring_params.enemy_distance - learning_rate * parameter_gradient.enemy_distance,
            enemy_stiffness=spring_params.enemy_stiffness - learning_rate * parameter_gradient.enemy_stiffness
        )

        print(f"friend_distance gradient: {parameter_gradient.friend_distance}")
        print(f"friend_stiffness gradient: {parameter_gradient.friend_stiffness}")
        print(f"enemy_distance gradient: {parameter_gradient.enemy_distance}")
        print(f"enemy_stiffness gradient: {parameter_gradient.enemy_stiffness}")

        print(f"friend_distance: {spring_params.friend_distance}")
        print(f"friend_stiffness: {spring_params.friend_stiffness}")
        print(f"enemy_distance: {spring_params.enemy_distance}")
        print(f"enemy_stiffness: {spring_params.enemy_stiffness}")

        metrics = evaluate(
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
    
    spring_state = init_spring_state(
        rng=rng,
        n=data.num_nodes,
        m=data.num_edges,
        embedding_dim=embedding_dim,
    )

    training_signs = signs.copy()
    training_signs = training_signs.at[train_mask].set(0)

    simulation_params.iterations = iterations

    spring_state = simulate(
        simulation_params,
        spring_state, 
        spring_params,
        attention_head,
        attention_head_params,
        edge_index,
        training_signs)

    metrics = evaluate(
        spring_state,
        edge_index,
        signs,
        train_mask,
        test_mask)

    print(metrics)

    # create four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    embeddings = spring_state.position
    # plot the embeddings
    ax1.scatter(embeddings[:, 0], embeddings[:, 1])# c=spring_state.energy)
    # color bar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax1)
    ax1.set_title('Embeddings')

    # # plot energies
    # ax2.hist(spring_state.energy)
    # ax2.set_title('Energies')

    # plot the energies over time, log scale
    ax3.plot(total_energies)
    ax3.set_yscale('log')
    ax3.set_title('Total energy')

    # plot measures
    ax4.plot(aucs)
    ax4.plot(f1_binaries)
    ax4.plot(f1_micros)
    ax4.plot(f1_macros)
    ax4.set_title('Measures')
    ax4.legend(['AUC', 'F1 binary', 'F1 micro', 'F1 macro'])

    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])
    