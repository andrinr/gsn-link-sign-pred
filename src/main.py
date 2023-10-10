# External dependencies
import sys, getopt
import torch_geometric.transforms as T
from torch_geometric.utils import is_undirected
import yaml
import inquirer
from jax import random, value_and_grad
import jax.numpy as jnp
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import tqdm

# Local dependencies
from data import Slashdot, BitcoinO, BitcoinA, WikiRFA, Epinions
from graph import train_test_val
from springs import SpringParams, init_spring_state, simulate

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
    iterations = 100
    time_step =  0.01
    damping = 0.2
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

    # Create train and test datasets
    data, training_mask, validation_mask, test_mask = train_test_val(data, 0.1, 0.8)
    training_mask = jnp.array(training_mask)
    validation_mask = jnp.array(validation_mask)
    test_mask = jnp.array(test_mask)
    
    # convert to jnp arrays from torch tensors
    edge_index = jnp.array(data.edge_index)
    signs = jnp.array(data.edge_attr)

    training_signs = signs.copy()
    training_signs = training_signs.at[~training_mask].set(0)

    spring_params = SpringParams(
        friend_distance=1.0,
        friend_stiffness=1.0,
        enemy_distance=1.0,
        enemy_stiffness=1.0,
        time_step=time_step,
        damping=damping,
    )
    
    rng = random.PRNGKey(42)

    logreg = LogisticRegression()
    total_energies = []
    aucs = []
    f1_binaries = []
    f1_micros = []
    f1_macros = []
    
    grad = value_and_grad(simulate, argnums=2, has_aux=True)
    
    learning_rate = 0.04
    for i in tqdm.trange(20):
        spring_state = init_spring_state(
            rng=rng,
            n=data.num_nodes,
            embedding_dim=embedding_dim,
        )
            
        (loss_value, spring_state), parameter_gradient = grad(
            iterations,
            spring_state, 
            spring_params,
            training_signs,
            edge_index)
        
        print(f"loss: {loss_value}")

        # update spring state
        spring_params = spring_params._replace(
            friend_distance=spring_params.friend_distance - learning_rate * parameter_gradient.friend_distance,
            friend_stiffness=spring_params.friend_stiffness - learning_rate * parameter_gradient.friend_stiffness,
            enemy_distance=spring_params.enemy_distance - learning_rate * parameter_gradient.enemy_distance,
            enemy_stiffness=spring_params.enemy_stiffness - learning_rate * parameter_gradient.enemy_stiffness,
            damping=spring_params.damping - learning_rate * parameter_gradient.damping,
            time_step=0.1,
        )

        print(f"friend_distance: {spring_params.friend_distance}")
        print(f"friend_stiffness: {spring_params.friend_stiffness}")
        print(f"enemy_distance: {spring_params.enemy_distance}")
        print(f"enemy_stiffness: {spring_params.enemy_stiffness}")
        print(f"damping: {spring_params.damping}")

        embeddings = spring_state.position
        position_i = embeddings.at[edge_index[0]].get()
        position_j = embeddings.at[edge_index[1]].get()

        spring_vec = position_i - position_j
        spring_vec_norm = jnp.linalg.norm(spring_vec, axis=1)
        spring_vec_norm = jnp.expand_dims(spring_vec_norm, axis=1)

        logreg.fit(spring_vec_norm.at[training_mask].get(), signs.at[training_mask].get())
        y_pred = logreg.predict(spring_vec_norm.at[validation_mask].get())

        auc = roc_auc_score(signs.at[validation_mask].get(), y_pred)
        f1_binary = f1_score(signs.at[validation_mask].get(), y_pred, average='binary')
        f1_micro = f1_score(signs.at[validation_mask].get(), y_pred, average='micro')
        f1_macro = f1_score(signs.at[validation_mask].get(), y_pred, average='macro')

        aucs.append(auc)
        f1_binaries.append(f1_binary)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)


    spring_state = init_spring_state(
        rng=rng,
        n=data.num_nodes,
        embedding_dim=embedding_dim,
    )

    loss, spring_state = simulate(
        iterations,
        spring_state, 
        spring_params,
        signs,
        edge_index)

    embeddings = spring_state.position

    position_i = embeddings.at[edge_index[0]].get()
    position_j = embeddings.at[edge_index[1]].get()

    spring_vec = position_i - position_j
    spring_vec_norm = jnp.linalg.norm(spring_vec, axis=1)
    spring_vec_norm = jnp.expand_dims(spring_vec_norm, axis=1)

    logreg.fit(spring_vec_norm.at[training_mask].get(), signs.at[training_mask].get())
    y_pred = logreg.predict(spring_vec_norm.at[test_mask].get())
    
    auc = roc_auc_score(signs.at[test_mask].get(), y_pred)
    f1_binary = f1_score(signs.at[test_mask].get(), y_pred, average='binary')
    f1_micro = f1_score(signs.at[test_mask].get(), y_pred, average='micro')
    f1_macro = f1_score(signs.at[test_mask].get(), y_pred, average='macro')

    aucs.append(auc)
    f1_binaries.append(f1_binary)
    f1_micros.append(f1_micro)
    f1_macros.append(f1_macro)

    print(f"auc: {auc}")
    print(f"f1_micro: {f1_micro}")
    print(f"f1_macro: {f1_macro}")
    print(f"f1_binary: {f1_binary}")

    # create four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # plot the embeddings
    ax1.scatter(embeddings[:, 0], embeddings[:, 1], c=spring_state.energy)
    # color bar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax1)
    ax1.set_title('Embeddings')

    # plot energies
    ax2.hist(spring_state.energy)
    ax2.set_title('Energies')

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
    