# External dependencies
import hydra
from omegaconf import DictConfig
from functools import partial
# torch imports
import torch_geometric.transforms as T
from torch_geometric.datasets import BitcoinOTC
# Local dependencies
from data import SignedDataset, BSCLGraph, even_exponential
from model import SignDenoising, Training
from visualize import visualize

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    # Define the model 
    model = SignDenoising(16, node_attr_size)
    training = Training(
        cfg=cfg,
        model=model)

    # Load the dataset
    if cfg.dataset.id == "bscl":
        degree_generator = partial(even_exponential, size=cfg.dataset.simulation.n_nodes, scale=5.0)
        BSCL_graph_kwargs = {
            "degree_generator": degree_generator,
            "p_positive_sign": cfg.dataset.simulation.p_positive,
            "p_close_triangle": cfg.simuldataset.simulation.BSCL.p_close_triangle,
            "p_close_for_balance": cfg.dataset.simulation.BSCL.p_close_for_balance,
            "remove_self_loops": cfg.dataset.simulation.BSCL.remove_self_loops
        }

        # We transform the graph to a line graph, meaning each edge is replaced with a node
        # Source: Line Graph Neural Networks for Link Prediction
        # Signs are now node features
        node_attr_size = cfg.dataset.pe_size + 1
        transform = T.Compose([
            T.LargestConnectedComponents(),
            T.LineGraph(force_directed=True),
            T.AddLaplacianEigenvectorPE(k=cfg.dataset.pe_size, attr_name='pe'),
            #T.AddRandomWalkPE(cfg.dataset.pe_size, attr_name='pe'),
            ])

        print("Loading dataset")
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

    elif cfg.dataset.id == "bitcoin":
        transform = T.Compose([
            T.LargestConnectedComponents(),
            T.LineGraph(force_directed=True),
            T.AddLaplacianEigenvectorPE(k=cfg.dataset.pe_size, attr_name='pe'),
            #T.AddRandomWalkPE(cfg.dataset.pe_size, attr_name='pe'),
            ])

        train_dataset = BitcoinOTC(root=cfg.dataset.root, transform=transform)
        # in this case node masks are used to split the dataset
        test_dataset = train_dataset
        use_node_mask = cfg.dataset.bitcoin.node_mask

        print(train_dataset[0].train_mask)

    training.train(dataset=train_dataset, epochs=10, use_node_mask=use_node_mask)
    training.test(dataset=test_dataset, use_node_mask=use_node_mask)

if __name__ == "__main__":
    main()