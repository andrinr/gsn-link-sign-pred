# External dependencies
import sys, getopt
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import is_undirected
import yaml
import inquirer
from jax import random
import jax.numpy as jnp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Local dependencies
from data import Slashdot, BitcoinO, BitcoinA, WikiRFA, Epinions
from graph import train_test_split
from springs import SpringParams, LogRegState, init_log_reg_state, update, train, predict, init_spring_state

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
    iterations = 400
    time_step =  0.01
    damping = 0.05
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

    optimizer_iterations = 0
    opts, args = getopt.getopt(argv,"s:h:d:i:o:p:",
        ["embedding_size=","time_step=", "damping=", "iterations=", "optimize="])
    for opt, arg in opts:
        if opt == '-s':
            embedding_dim = int(arg)
        elif opt == '-h':
            time_step = int(arg)
        elif opt == '-d':
            damping = int(arg)
        elif opt == '-i':
            iterations = int(arg)
        elif opt == '-o':
            optimizer_iterations = int(arg)

    if optimizer_iterations == 0 :
        stream = open("src/params.yaml", 'r')
        params = yaml.load(stream, Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pre_transform = T.Compose([
        #CycleTransform(max_degree=8)
    ])

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

    for i in range(iterations):
        spring_state = update(
            state=spring_state,
            params=spring_params,
            sign=signs,
            edge_index=edge_index,
        )

    embeddings = spring_state.position

    position_i = embeddings.at[edge_index[0]].get()
    position_j = embeddings.at[edge_index[1]].get()

    spring_vector = position_i - position_j
    spring_vector_norm = jnp.linalg.norm(spring_vector, axis=1)

    logreg_state = init_log_reg_state()

    for i in range(1000):
        logreg_state, loss = train(
            state=logreg_state,
            rate=0.01,
            X=spring_vector_norm.at[training_mask == 1].get(),
            y=signs.at[training_mask == 1].get(),
        )

    y_pred = predict(
        state=logreg_state,
        X=spring_vector_norm.at[training_mask == 0].get(),
    )

    auc = roc_auc_score(signs.at[training_mask == 0].get(), y_pred)
    f1_micro = f1_score(signs.at[training_mask == 0].get(), y_pred, average='micro')
    f1_macro = f1_score(signs.at[training_mask == 0].get(), y_pred, average='macro')
    f1_binary = f1_score(signs.at[training_mask == 0].get(), y_pred, average='binary')

    print(f"auc: {auc}")
    print(f"f1_micro: {f1_micro}")
    print(f"f1_macro: {f1_macro}")
    print(f"f1_binary: {f1_binary}")

    print(f"confusion matrix: {confusion_matrix(training_mask, y_pred)}")
    

if __name__ == "__main__":
    main(sys.argv[1:])
    