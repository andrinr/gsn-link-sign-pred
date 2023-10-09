# External dependencies
import nevergrad as ng
import sys, getopt
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.utils import is_undirected
import yaml
import inquirer
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from networkx.algorithms.cycles import simple_cycles

# Local dependencies
from springs import Training, Embeddings
from data import Slashdot, BitcoinO, BitcoinA, WikiRFA, Epinions
from stats import Edges, Section
from graph import CycleTransform, train_test_split

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
    
    training_mask = training_data.edge_attr != 0
    
    # edge_sampling = Edges(data, n_edges=3000, mask=test_mask)
    # edge_sampling()
    training_data = training_data.to(device)
    test_data = test_data.to(device)

    embeddings = Embeddings(
        edge_index=training_data.edge_index,
        signs=training_data.edge_attr,
        training_mask=training_mask,
        embedding_dim=embedding_dim,
        time_step=time_step,
        iterations=iterations,
        damping=damping,
        friend_distance=5.0,
        friend_stiffness=5.0,
        neutral_distance=params['neutral_distance'],
        neutral_stiffness=params['neutral_stiffness'],
        enemy_distance=params['enemy_distance'],
        enemy_stiffness=params['enemy_stiffness'],
    )
    
    pos, aucs, f1_binaries, f1_micros, f1_macros = embeddings(num_intervals=1)

    print("AUC")
    print(aucs)

    # confusion_matrix, part_of_balanced, part_of_unbalanced = edge_sampling.compare(training.y_pred)

    # print("Confusion matrix")
    # print(confusion_matrix)
    # print("Part of balanced")
    # print(part_of_balanced)
    # print("Part of unbalanced")
    # print(part_of_unbalanced)

if __name__ == "__main__":
    main(sys.argv[1:])
    