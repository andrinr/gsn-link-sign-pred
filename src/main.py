# External dependencies
import nevergrad as ng
import sys, getopt
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.utils import is_undirected
import yaml
import inquirer
# Local dependencies
from model import Training
from data import Slashdot, BitcoinO, BitcoinA, WikiRFA, Epinions
from stats import Triplets
from graph import train_test_split, order_shuffle

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
    iterations = 500
    time_step =  0.005
    damping = 0.02
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

    plot_stats = False

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
        elif opt == '-p':
            plot_stats = True

    if optimizer_iterations == 0 :
        stream = open("src/params.yaml", 'r')
        params = yaml.load(stream, Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset_name == "BitcoinOTC":
        dataset = BitcoinO(root= root)

    elif dataset_name == "Bitcoin_Alpha":
        dataset = BitcoinA(root= root)
    
    elif dataset_name == "WikiRFA":
        dataset = WikiRFA(root= root)
        
    elif dataset_name == "Slashdot":
        dataset = Slashdot(root= root)
        
    elif dataset_name == "Epinions":
        dataset = Epinions(root= root)

    data = dataset[0]
    # if not is_undirected(data.edge_index):
    #     transform = T.ToUndirected(reduce="min")
    #     data = transform(data)

    # Create train and test datasets
    data, training_data, test_data = train_test_split(
        data = data, 
        train_percentage=0.8)
    
    assert data.is_undirected()

    stats = Triplets(data)
    stats.sample(1000)
    stats.stats()
    print(f"p_balanced: {stats.p_balanced}")

    n_edges = data.edge_attr.shape[0]

    print(f"Number of edges: {n_edges}")
    print(f"Number of nodes: {data.num_nodes}")

    training = Training(
        device=device,
        train_data=training_data,
        test_data=test_data,
        embedding_dim= embedding_dim,
        time_step= time_step,
        iterations= iterations,
        damping= damping,
        friend_distance=5.0,
        friend_stiffness=5.0)

    if optimizer_iterations > 0:
        # Instrumentation class is used for functions with multiple inputs
        # (positional and/or keywords)
        parametrization = ng.p.Instrumentation(
            neutral_distance=ng.p.Scalar(lower=0.1, upper=30.0),
            neutral_stiffness=ng.p.Scalar(lower=0.1, upper=8.0),
            enemy_distance=ng.p.Scalar(lower=0.1, upper=30),
            enemy_stiffness=ng.p.Scalar(lower=0.1, upper=8.0),
        )

        optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=optimizer_iterations, num_workers=1)

        recommendation = optimizer.minimize(training)

        # convert to dict
        recommendation = dict(recommendation.kwargs)
        print(recommendation)

        user_input = input('Do you want to store the preferences? (y/n) ')

        if user_input.lower() == 'y':
            with open('src/params.yaml', 'w') as outfile:
                yaml.dump(recommendation, outfile, default_flow_style=False)

    else:
        
        training(
            neutral_distance= params['neutral_distance'],
            neutral_stiffness= params['neutral_stiffness'],
            enemy_distance= params['enemy_distance'],
            enemy_stiffness= params['enemy_stiffness'],
        )

    test_mask = training_data.edge_attr == 0
    sbc, suc, dbc, duc = stats.compare(training.y_pred, test_mask)
    
    print(f"Single balanced correct: {sbc}")
    print(f"Single unbalanced correct: {suc}")
    print(f"Double balanced correct: {dbc}")
    print(f"Double unbalanced correct: {duc}")

if __name__ == "__main__":
    main(sys.argv[1:])
    