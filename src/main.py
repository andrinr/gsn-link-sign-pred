# External dependencies
import hydra
from omegaconf import DictConfig
from functools import partial
import networkx as nx
import matplotlib.pyplot as plt
# torch imports
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
# Local dependencies
from data import SignedDataset, BSCLGraph, even_exponential
from model import SpringTransform, log_regression
from data import WikiSigned, Tribes, Chess, BitcoinA
from torch_geometric.transforms import RandomLinkSplit

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define the transforms
    pre_transforms = []
    pre_transforms.append(T.ToUndirected())
    if cfg.dataset.transform.largest_cc:
        pre_transforms.append(T.LargestConnectedComponents())

    if cfg.dataset.transform.line_graph:
        pre_transforms.append(T.LineGraph(force_directed=True))
    #transform.append(T.ToSparseTensor())
    pre_transforms = T.Compose(pre_transforms)

    # Define the dataset
    print("Loading dataset")
    if cfg.dataset.id == "bscl":
        degree_generator = partial(even_exponential, size=cfg.dataset.simulation.n_nodes, scale=5.0)
        BSCL_graph_kwargs = {
            "degree_generator": degree_generator,
            "p_positive_sign": cfg.dataset.simulation.p_positive,
            "p_close_triangle": cfg.dataset.simulation.BSCL.p_close_triangle,
            "p_close_for_balance": cfg.dataset.simulation.BSCL.p_close_for_balance,
            "remove_self_loops": cfg.dataset.simulation.BSCL.remove_self_loops
        }

        dataset = SignedDataset(
            graph_generator=BSCLGraph,
            graph_generator_kwargs=BSCL_graph_kwargs,
            transform=pre_transforms,
            num_graphs=int(cfg.dataset.simulation.n_simulations * cfg.dataset.train_size))

    elif cfg.dataset.id == "wiki":

        dataset = WikiSigned(
            root=cfg.dataset.root,
            pre_transform=pre_transforms,
            one_hot_signs=False)

    elif cfg.dataset.id == "tribes":
        dataset = Tribes(
            root=cfg.dataset.root,
            pre_transform=pre_transforms,
            one_hot_signs=False)

    elif cfg.dataset.id == "chess":
        dataset = Chess(
            root=cfg.dataset.root,
            pre_transform=pre_transforms,
            one_hot_signs=False)

    elif cfg.dataset.id == "bitcoin":
        dataset = BitcoinA(
            root=cfg.dataset.root,
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
        np.random.choice([1, 0], size=n_edges, p=[cfg.dataset.test_size, 1-cfg.dataset.test_size])
    test_mask = torch.tensor(test_mask)

    train_data.edge_attr = torch.where(test_mask == 1, 0, dataset[0].edge_attr)
    test_data.edge_attr = torch.where(test_mask == 0, 0, dataset[0].edge_attr)

    springTransform = SpringTransform(
        device=device,
        embedding_dim=cfg.model.spring_pe.embedding_dim,
        time_step=cfg.model.spring_pe.step_size,
        iterations=cfg.model.spring_pe.iterations,
        damping=cfg.model.spring_pe.damping,
        friend_distance=cfg.model.spring_pe.friend_distance,
        friend_stiffness=cfg.model.spring_pe.friend_stiffness,
        neutral_distance=cfg.model.spring_pe.neutral_distance,
        neutral_stiffness=cfg.model.spring_pe.neutral_stiffness,
        enemy_distance=cfg.model.spring_pe.enemy_distance,
        enemy_stiffness=cfg.model.spring_pe.enemy_stiffness,
    )

    train_data = springTransform(train_data)

    auc, acc, prec, rec, f1 = log_regression(train_data, test_data, test_mask )

    print(f"AUC: {auc}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1: {f1}")

if __name__ == "__main__":
    main()