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

# Local dependencies
from model import Training
from data import Slashdot, BitcoinO, BitcoinA, WikiRFA, Epinions
from stats import Triplets
from graph import train_test_split

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

    match dataset_name:
        case "BitcoinOTC":
            dataset = BitcoinO(root= root)
        case "Bitcoin_Alpha":
            dataset = BitcoinA(root= root)
        case "WikiRFA":
            dataset = WikiRFA(root= root)
        case "Slashdot":
            dataset = Slashdot(root= root)
        case "Epinions":
            dataset = Epinions(root= root)

    data = dataset[0]
    if not is_undirected(data.edge_index):
        transform = T.ToUndirected(reduce="min")
        data = transform(data)

    # Create train and test datasets
    data, training_data, test_data = train_test_split(
        data = data, 
        train_percentage=0.8)
    
    stats = Triplets(data)
    stats.sample(6000)
    stats.stats()

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

    test_mask = training_data.edge_attr == 0
    total_balanced, correct_balanced, total_unbalanced, correct_unbalanced =\
        stats.compare(training.y_pred, test_mask)

    correct = np.concatenate((correct_balanced / total_balanced , correct_unbalanced / total_unbalanced))
    df = pd.DataFrame({
        'neutral': [1, 2, 3, 1, 2, 3],
        'balanced': [1, 1, 1, 0, 0, 0],
        'correct': correct
    })
    
    sns.catplot(
        data=df, kind="bar",
        x="neutral", y="correct", hue="balanced"
    )
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
    