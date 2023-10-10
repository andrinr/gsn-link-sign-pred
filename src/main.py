# External dependencies
import sys, getopt
import torch_geometric.transforms as T
from torch_geometric.utils import is_undirected
import yaml
import inquirer
from jax import random
import jax.numpy as jnp
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import tqdm

# Local dependencies
from data import Slashdot, BitcoinO, BitcoinA, WikiRFA, Epinions
from graph import train_test_split
from springs import SpringParams, init_log_reg_state, update, train, predict, init_spring_state

def main(argv) -> None:
    """
    Main function

    Parameters:
    ----------  
    -s : int (default=64)
        Embedding dimension
    -h : float (default=0.005)
        Time step
    -d : float (default=0.02)
        Damping
    -i : int (default=500)
        Number of iterations
    -o : int (default=0)
        Number of iterations for the optimizer
    """
    embedding_dim = 64
    iterations = 1500
    time_step =  0.01
    damping = 0.01
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

    opts,_ = getopt.getopt(argv,"s:h:d:i:p:",
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
    data, training_data, test_data = train_test_split(
        data = data, 
        train_percentage=0.8)
    
    # convert to jnp arrays from torch tensors
    edge_index = jnp.array(data.edge_index)
    signs = jnp.array(data.edge_attr)

    training_mask = training_data.edge_attr != 0
    training_mask = jnp.array(training_mask)

    training_signs = signs.copy()
    training_signs = training_signs.at[training_mask == 1].set(0)

    spring_params = SpringParams(
        friend_distance=5.0,
        friend_stiffness=5.0,
        neutral_distance=params['neutral_distance'],
        neutral_stiffness=params['neutral_stiffness'],
        enemy_distance=params['enemy_distance'],
        enemy_stiffness=params['enemy_stiffness'],
        time_step=time_step,
        damping=damping,
    )
    
    rng = random.PRNGKey(42)

    spring_state = init_spring_state(
        rng=rng,
        n=data.num_nodes,
        embedding_dim=embedding_dim,
    )

    logreg = LogisticRegression()
    total_energies = []
    aucs = []
    f1_binaries = []
    f1_micros = []
    f1_macros = []

    iter = tqdm.trange(iterations)
    for i in iter:
        spring_state = update(
            state=spring_state,
            params=spring_params,
            sign=training_signs,
            edge_index=edge_index,
        )

        total_energy = jnp.sum(spring_state.energy)
        total_energies.append(total_energy)
        # format the energy to 3 decimal places
        iter.set_description(f"Energy: {total_energy:.3f}")

        if i % 100 == 0:
            embeddings = spring_state.position
            position_i = embeddings.at[edge_index[0]].get()
            position_j = embeddings.at[edge_index[1]].get()

            spring_vec = position_i - position_j
            spring_vec_norm = jnp.linalg.norm(spring_vec, axis=1)
            spring_vec_norm = jnp.expand_dims(spring_vec_norm, axis=1)

            logreg.fit(spring_vec_norm.at[training_mask == 1].get(), signs.at[training_mask == 1].get())
            y_pred = logreg.predict(spring_vec_norm.at[training_mask == 0].get())

            auc = roc_auc_score(signs.at[training_mask == 0].get(), y_pred)
            f1_binary = f1_score(signs.at[training_mask == 0].get(), y_pred, average='binary')
            f1_micro = f1_score(signs.at[training_mask == 0].get(), y_pred, average='micro')
            f1_macro = f1_score(signs.at[training_mask == 0].get(), y_pred, average='macro')

            aucs.append(auc)
            f1_binaries.append(f1_binary)
            f1_micros.append(f1_micro)
            f1_macros.append(f1_macro)


    embeddings = spring_state.position

    position_i = embeddings.at[edge_index[0]].get()
    position_j = embeddings.at[edge_index[1]].get()

    spring_vec = position_i - position_j
    spring_vec_norm = jnp.linalg.norm(spring_vec, axis=1)
    spring_vec_norm = jnp.expand_dims(spring_vec_norm, axis=1)

    logreg = LogisticRegression()

    # logreg_state = init_log_reg_state()

    # losses = []
    # for _ in range(1000):
    #     logreg_state, loss = train(
    #         state=logreg_state,
    #         rate=0.01,
    #         X=spring_vector_norm.at[training_mask == 1].get(),
    #         y=signs.at[training_mask == 1].get(),
    #     )
    #     losses.append(loss)

    # y_pred = predict(
    #     state=logreg_state,
    #     X=spring_vector_norm.at[training_mask == 0].get(),
    # )

    # signs = signs > 0

    print(f"auc: {aucs[-1]}")
    print(f"f1_micro: {f1_micros[-1]}")
    print(f"f1_macro: {f1_macros[-1]}")
    print(f"f1_binary: {f1_binaries[-1]}")

    # create four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # plot the embeddings
    ax1.scatter(embeddings[:, 0], embeddings[:, 1])
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
    