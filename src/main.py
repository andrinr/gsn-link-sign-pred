# External dependencies
import hydra
from omegaconf import DictConfig
from functools import partial
import networkx as nx
import matplotlib.pyplot as plt
# torch imports
import torch
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
    
    pre_transforms = RandomLinkSplit(is_undirected=True, num_val=0.05, num_test=0.2, split_labels=False)
    train_data, val_data, test_data = pre_transforms(dataset[0])

    print(dataset[0])
    print(train_data.edge_label)
    print(train_data.edge_label_index)
    print(train_data)
    print(test_data)
    print(val_data)

    print(train_data.edge_attr)
    print(test_data.edge_attr)
    print((train_data.edge_index == val_data.edge_index).sum().item())
    print(dataset[0].edge_index.shape)
    print(train_data.edge_index.shape)
    print(test_data.edge_index.shape)

    springTransform = SpringTransform(
        device=device,
        embedding_dim=cfg.model.spring_pe.embedding_dim,
        time_step=cfg.model.spring_pe.step_size,
        stiffness=cfg.model.spring_pe.stiffness,
        iterations=cfg.model.spring_pe.iterations,
        damping=cfg.model.spring_pe.damping,
        noise=cfg.model.spring_pe.noise,
        friend_distance=cfg.model.spring_pe.friend_distance,
        enemy_distance=cfg.model.spring_pe.enemy_distance,
    )

    train_data = springTransform(train_data)

    acc, prec, rec, f1 = log_regression(train_data, test_data)

    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1: {f1}")

if __name__ == "__main__":
    main()