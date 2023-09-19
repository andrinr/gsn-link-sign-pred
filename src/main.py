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
from springs import Training
from data import Slashdot, BitcoinO, BitcoinA, WikiRFA, Epinions
from stats import Edges
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
    embedding_dim = 32
    iterations = 500
    time_step =  0.005
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
    
    test_mask = training_data.edge_attr == 0
    
    # edge_sampling = Edges(data, n_edges=3000, mask=test_mask)
    # edge_sampling()
    
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
        parametrization = ng.p.Instrumentation(
            neutral_distance=ng.p.Scalar(lower=0.1, upper=30.0),
            neutral_stiffness=ng.p.Scalar(lower=0.1, upper=8.0),
            enemy_distance=ng.p.Scalar(lower=0.1, upper=30),
            enemy_stiffness=ng.p.Scalar(lower=0.1, upper=8.0),
        )

        optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=optimizer_iterations, num_workers=1)

        recommendation = optimizer.minimize(training)
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

    # find number of correct and incorrect edges per node
    test_mask = training_data.edge_attr == 0
    correct_edges = data.edge_attr == training.y_pred
    incorrect_edges = data.edge_attr != training.y_pred

    correct_edges[~test_mask] = True

    correct_edges_per_node = torch.zeros(training_data.num_nodes)
    incorrect_edges_per_node = torch.zeros(training_data.num_nodes)
    n_edges_per_node = torch.zeros(training_data.num_nodes)

    for i in range(training_data.num_nodes):
        edge_indices = torch.where(training_data.edge_index[0] == i)[0]
        correct_edges_per_node[i] = correct_edges[edge_indices].sum()

        incorrect_edges_per_node[i] = incorrect_edges[edge_indices].sum()

        n_edges_per_node[i] = len(edge_indices)

    # 4 subplots
    fig, axs = plt.subplots(2, 2)

    ratio = correct_edges_per_node / (incorrect_edges_per_node + correct_edges_per_node)
    
    axs[0, 0].scatter(
        ratio,
        training.final_per_node_force,
        s=n_edges_per_node)
    
    axs[0, 0].set_xlabel('Percentage of correct edges')
    axs[0, 0].set_ylabel('Force')

    axs[0, 1].scatter(
        ratio,
        training.final_per_node_energy,
        s=n_edges_per_node)
    
    axs[0, 1].set_xlabel('Percentage of correct edges')
    axs[0, 1].set_ylabel('Energy')

    axs[1, 0].scatter(
        ratio,
        training.final_per_node_vel,
        s=n_edges_per_node)
    
    axs[1, 0].set_xlabel('Percentage of correct edges')
    axs[1, 0].set_ylabel('Velocity')

    axs[1, 1].scatter(
        ratio,
        training.final_per_node_pos,
        s=n_edges_per_node)
    
    axs[1, 1].set_xlabel('Percentage of correct edges')
    axs[1, 1].set_ylabel('Position')

    plt.show()

    # confusion_matrix, part_of_balanced, part_of_unbalanced = edge_sampling.compare(training.y_pred)

    # print("Confusion matrix")
    # print(confusion_matrix)
    # print("Part of balanced")
    # print(part_of_balanced)
    # print("Part of unbalanced")
    # print(part_of_unbalanced)

if __name__ == "__main__":
    main(sys.argv[1:])
    