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
from model import OpinionEmbedding, SignDenoising2, Training
from data import WikiSigned, Tribes
#from pyg_nn.models import DGCNN

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define the transforms
    transform = []
    if cfg.dataset.transform.largest_cc:
        transform.append(T.LargestConnectedComponents())

    if cfg.dataset.transform.line_graph:
        transform.append(T.LineGraph(force_directed=True))
  
    transform.append(OpinionEmbedding(
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
    transform = T.Compose(transform)

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

        train_dataset = SignedDataset(
            graph_generator=BSCLGraph,
            graph_generator_kwargs=BSCL_graph_kwargs,
            transform=transform,
            num_graphs=int(cfg.dataset.simulation.n_simulations * cfg.dataset.train_size))

        test_dataset = SignedDataset(
            graph_generator=BSCLGraph,
            graph_generator_kwargs=BSCL_graph_kwargs,
            transform=transform,
            num_graphs=int(cfg.dataset.simulation.n_simulations * ( 1 - cfg.dataset.train_size)))
        use_node_mask = cfg.dataset.simulation.BSCL.node_mask

    elif cfg.dataset.id == "wiki":

        train_dataset = WikiSigned(
            root=cfg.dataset.root,
            pre_transform=transform,
            one_hot_signs=False)
        # in this case node masks are used to split the dataset
        test_dataset = train_dataset

    elif cfg.dataset.id == "tribes":
        train_dataset = Tribes(
            root=cfg.dataset.root,
            pre_transform=transform,
            one_hot_signs=False)
        # in this case node masks are used to split the dataset
        test_dataset = train_dataset

    print(train_dataset[0])
    input_channels = train_dataset[0].x.shape[1]
    hidden_channels = cfg.model.hidden_channels
    output_channels = 2
    """peModel = DGCNN(emb_size=64)

    training_pe = Training(
        cfg=cfg,
        model=peModel,
        offset_unbalanced=False
    )"""


    # iterate over each edge in the pyg graph
    for i, j in train_dataset[0].edge_index.t().tolist():
        distance = torch.norm(train_dataset[0].x[i] - train_dataset[0].x[j])
        edge_indices = torch.where((train_dataset[0].edge_index[0] == i) & (train_dataset[0].edge_index[1] == j))
        actual_sign = train_dataset[0].edge_attr[edge_indices]
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
    training.train(dataset=train_dataset, epochs=cfg.model.epochs)
    training.test(dataset=test_dataset)

if __name__ == "__main__":
    main()