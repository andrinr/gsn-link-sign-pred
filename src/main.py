# External dependencies
import hydra
from omegaconf import DictConfig
import nevergrad as ng
# torch imports
import torch
import numpy as np
import torch_geometric.transforms as T
import torch_geometric.data as Data
from model import Training
from concurrent import futures
# Local dependencies
from data import WikiSigned, Tribes, Chess, BitcoinA

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define the transforms
    pre_transforms = []
    pre_transforms.append(T.ToUndirected())
    #transform.append(T.ToSparseTensor())
    pre_transforms = T.Compose(pre_transforms)

    if cfg.id == "wiki":

        dataset = WikiSigned(
            root=cfg.root,
            pre_transform=pre_transforms,
            one_hot_signs=False)

    elif cfg.id == "tribes":
        dataset = Tribes(
            root=cfg.root,
            pre_transform=pre_transforms,
            one_hot_signs=False)

    elif cfg.id == "chess":
        dataset = Chess(
            root=cfg.root,
            pre_transform=pre_transforms,
            one_hot_signs=False)

    elif cfg.id == "bitcoin":
        dataset = BitcoinA(
            root=cfg.root,
            pre_transform=pre_transforms)
    
    data = dataset[0]
    n_edges = data.edge_index.shape[1]

    print(f"Number of edges: {n_edges}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of positive edges: {data.edge_attr.sum()}")
    print(f"Number of negative edges: {n_edges - data.edge_attr.sum()}")
    print(f"Ratio of positive edges: {data.edge_attr.sum() / n_edges}")

    # Create train and test datasets
    train_data = dataset[0].clone()
    test_data = dataset[0].clone()
    test_mask =\
        np.random.choice([1, 0], size=n_edges, p=[cfg.test_size, 1-cfg.test_size])
    test_mask = torch.tensor(test_mask)

    train_data.edge_attr = torch.where(test_mask == 1, 0, dataset[0].edge_attr)
    test_data.edge_attr = torch.where(test_mask == 0, 0, dataset[0].edge_attr)

    training = Training(
        device=device,
        train_data=train_data,
        test_data=test_data,
        test_mask=test_mask,
        embedding_dim=cfg.embedding_dim,
        time_step=cfg.time_step,
        iterations=cfg.iterations,
        damping=cfg.damping)

    # Instrumentation class is used for functions with multiple inputs
    # (positional and/or keywords)
    parametrization = ng.p.Instrumentation(
        friend_distance=ng.p.Scalar(lower=0.1, upper=10.0),
        friend_stiffness=ng.p.Scalar(lower=0.1, upper=8.0),
        neutral_distance=ng.p.Scalar(lower=0.1, upper=10.0),
        neutral_stiffness=ng.p.Scalar(lower=0.1, upper=8.0),
        enemy_distance=ng.p.Scalar(lower=0.1, upper=10),
        enemy_stiffness=ng.p.Scalar(lower=0.1, upper=8.0),
    )

    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=100, num_workers=1)

  
    recommendation = optimizer.minimize(training)

    print(recommendation.args)
    print(recommendation.value)

if __name__ == "__main__":
    main()