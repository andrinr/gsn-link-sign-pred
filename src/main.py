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
from data import WikiSigned, Slashdot, BitcoinO, BitcoinA, Tribes, WikiRFA
def main(argv) -> None:
    embedding_dim = 64
    iterations = 500
    time_step =  0.005
    damping = 0.02
    root = 'src/data/'
    test_size = 0.2

    questions = [
    inquirer.List('dataset',
                    message="Choose a dataset",
                    choices=[
                        'Bitcoin_Alpha', 
                        'BitcoinOTC', 
                        'WikiRFA', 
                        'Tribes'],
                ),
    ]
    answers = inquirer.prompt(questions)
    dataset_name = answers['dataset']


    optimizer_iterations = 0
    opts, args = getopt.getopt(argv,"s:h:d:i:o:",
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
    # Define the transforms
    pre_transforms = []
    #pre_transforms.append(T.ToUndirected(reduce="mean"))
    #transform.append(T.ToSparseTensor())
    pre_transforms = T.Compose(pre_transforms)

    if dataset_name == "wiki":
        dataset = WikiSigned(
            root= root,
            pre_transform=pre_transforms,
            one_hot_signs=False)

    elif dataset_name == 'Tribes':
        dataset = Tribes(
            root= root
        )

    elif dataset_name == "BitcoinOTC":
        dataset = BitcoinO(
            root= root,
            pre_transform=pre_transforms)

    elif dataset_name == "Bitcoin_Alpha":
        dataset = BitcoinA(
            root= root,
            pre_transform=pre_transforms)
    
    elif dataset_name == "WikiRFA":
        dataset = WikiRFA(
            root= root,
            pre_transform=pre_transforms)

    data = dataset[0]
    if not is_undirected(data.edge_index):
        print("is directed")
        # transform to directed graph
        #transform = T.ToUndirected(reduce="mean")
        #data = transform(data)

    n_edges = data.edge_attr.shape[0]

    print(f"Number of edges: {n_edges}")
    print(f"Number of nodes: {data.num_nodes}")
    # Create train and test datasets
    train_data = data.clone()
    test_data = data.clone()
    test_mask =\
        np.random.choice([1, 0], size=n_edges, p=[ test_size, 1- test_size])
    test_mask = torch.tensor(test_mask)

    train_data.edge_attr = torch.where(test_mask == 1, 0, train_data.edge_attr)
    test_data.edge_attr = torch.where(test_mask == 0, 0,  test_data.edge_attr)

    training = Training(
        device=device,
        train_data=train_data,
        test_data=test_data,
        test_mask=test_mask,
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
            neutral_distance=ng.p.Scalar(lower=0.1, upper=20.0),
            neutral_stiffness=ng.p.Scalar(lower=0.1, upper=8.0),
            enemy_distance=ng.p.Scalar(lower=0.1, upper=20),
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

if __name__ == "__main__":
    main(sys.argv[1:])