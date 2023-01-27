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
from model import SpringTransform
from data import WikiSigned, Tribes, Chess, BitcoinA
from torch_geometric.transforms import RandomLinkSplit
from sklearn.linear_model import LogisticRegression
#from pyg_nn.models import DGCNN

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define the transforms
    linkSplit = []
    if cfg.dataset.transform.largest_cc:
        linkSplit.append(T.LargestConnectedComponents())

    if cfg.dataset.transform.line_graph:
        linkSplit.append(T.LineGraph(force_directed=True))
  
    linkSplit.append(SpringTransform(
        device=device,
        embedding_dim=cfg.model.spring_pe.embedding_dim,
        time_step=cfg.model.spring_pe.step_size,
        stiffness=cfg.model.spring_pe.stiffness,
        iterations=cfg.model.spring_pe.iterations,
        damping=cfg.model.spring_pe.damping,
        noise=cfg.model.spring_pe.noise,
        friend_distance=cfg.model.spring_pe.friend_distance,
        enemy_distance=cfg.model.spring_pe.enemy_distance,
    ))

    #transform.append(T.ToSparseTensor())
    linkSplit = T.Compose(linkSplit)

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
            transform=linkSplit,
            num_graphs=int(cfg.dataset.simulation.n_simulations * cfg.dataset.train_size))

    elif cfg.dataset.id == "wiki":

        dataset = WikiSigned(
            root=cfg.dataset.root,
            pre_transform=linkSplit,
            one_hot_signs=False)

    elif cfg.dataset.id == "tribes":
        dataset = Tribes(
            root=cfg.dataset.root,
            pre_transform=linkSplit,
            one_hot_signs=False)

    elif cfg.dataset.id == "chess":
        dataset = Chess(
            root=cfg.dataset.root,
            pre_transform=linkSplit,
            one_hot_signs=False)

    elif cfg.dataset.id == "bitcoin":
        dataset = BitcoinA(
            root=cfg.dataset.root,
            pre_transform=linkSplit)
        
    linkSplit = RandomLinkSplit(is_undirected=True, num_val=0)
    train_data, val_data, test_data = linkSplit(dataset[0])

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

    for i, j in train_data.edge_index.t().tolist():
        x = test_data.x[i]
        y = test_data.x[j]

        # logistic regression classifier

    input_channels = dataset[0].x.shape[1]
    hidden_channels = cfg.model.hidden_channels
    output_channels = 2
    """peModel = DGCNN(emb_size=64)

    training_pe = Training(
        cfg=cfg,
        model=peModel,
        offset_unbalanced=False
    )"""


    # iterate over each edge in the pyg graph
    for i, j in dataset[0].edge_index.t().tolist():
        distance = torch.norm(dataset[0].x[i] - dataset[0].x[j])
        edge_indices = torch.where((dataset[0].edge_index[0] == i) & (dataset[0].edge_index[1] == j))
        actual_sign = dataset[0].edge_attr[edge_indices]
        print(f"Edge {i} -> {j} has distance {distance} and sign {actual_sign}")

    """peModel = DGCNN(emb_size=64)

    training_pe = Training(
        cfg=cfg,
        model=peModel,
        offset_unbalanced=False
    )"""

    # Define the model 
    #model = SignDenoising(16, node_attr_size)
    model2 = SignDenoising2(input_channels, hidden_channels, output_channels)

    training = Training(
        cfg=cfg.model,
        model=model2,
        device=device)

    # Train and test
    training.train(dataset=dataset, epochs=cfg.model.epochs)
    training.test(dataset=test_dataset)

if __name__ == "__main__":
    main()