# External dependencies
import nevergrad as ng
import sys, getopt
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.utils import is_undirected
import yaml
# Local dependencies
from model import Training
from data import WikiSigned, Tribes, Chess, BitcoinA, Epinions, WikiEdits

def main(argv) -> None:
    embedding_dim = 64
    iterations = 500
    time_step =  0.005
    damping = 0.02
    dataset_name = 'bitcoin'
    root = 'src/data/'
    test_size = 0.2

    optimizer_iterations = 0
    opts, args = getopt.getopt(argv,"s:h:d:i:o:",
        ["embedding_size=","time_step=", "damping=", "iterations=", "optimze="])
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
        stream = open("src/conf/params.yaml", 'r')
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

    elif dataset_name == "tribes":
        dataset = Tribes(
            root= root,
            pre_transform=pre_transforms,
            one_hot_signs=False)

    elif dataset_name == "chess":
        dataset = Chess(
            root= root,
            pre_transform=pre_transforms,
            one_hot_signs=False)

    elif dataset_name == "bitcoin":
        dataset = BitcoinA(
            root= root,
            pre_transform=pre_transforms)
    
    elif dataset_name == "epinions":
        dataset = Epinions(
            root= root,
            pre_transform=pre_transforms)
    
    elif dataset_name == "wikiedits":
        dataset = WikiEdits(
            root= root,
            pre_transform=pre_transforms)

    data = dataset[0]
    n_edges = data.edge_index.shape[1]

    print(f"Number of edges: {n_edges}")
    print(f"Number of nodes: {data.num_nodes}")
    print("is directed", not is_undirected(data.edge_index))
    # Create train and test datasets
    train_data = dataset[0].clone()
    test_data = dataset[0].clone()
    test_mask =\
        np.random.choice([1, 0], size=n_edges, p=[ test_size, 1- test_size])
    test_mask = torch.tensor(test_mask)

    train_data.edge_attr = torch.where(test_mask == 1, 0, train_data.edge_attr)
    test_data.edge_attr = torch.where(test_mask == 0, 0,  test_data.edge_attr)

    print(train_data.edge_attr)
    training = Training(
        device=device,
        train_data=train_data,
        test_data=test_data,
        test_mask=test_mask,
        embedding_dim= embedding_dim,
        time_step= time_step,
        iterations= iterations,
        damping= damping)

    if optimizer_iterations > 0:
        # Instrumentation class is used for functions with multiple inputs
        # (positional and/or keywords)
        parametrization = ng.p.Instrumentation(
            friend_distance=ng.p.Scalar(lower=0.1, upper=20.0),
            friend_stiffness=ng.p.Scalar(lower=0.1, upper=8.0),
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
        
        with open('src/conf/params.yaml', 'w') as outfile:
            yaml.dump(recommendation, outfile, default_flow_style=False)

    else:
        training(
            friend_distance= params['friend_distance'],
            friend_stiffness= params['friend_stiffness'],
            neutral_distance= params['neutral_distance'],
            neutral_stiffness= params['neutral_stiffness'],
            enemy_distance= params['enemy_distance'],
            enemy_stiffness= params['enemy_stiffness'],
        )
        

if __name__ == "__main__":
    main(sys.argv[1:])