# External dependencies
import hydra
from omegaconf import DictConfig
from functools import partial
# torch imports
import torch_geometric.transforms as T
# Local dependencies
from data import SignedDataset, BSCLGraph, even_exponential
from model import SignDenoising, SignDenoising2, Training
from visualize import visualize
from data import BitcoinOTC

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    node_attr_size = cfg.dataset.transform.pe_size + 1

    # Define the transforms
    transform = []
    if cfg.dataset.transform.largest_cc:
        transform.append(T.LargestConnectedComponents())

    if cfg.dataset.transform.line_graph:
        transform.append(T.LineGraph(force_directed=True))

    if cfg.dataset.transform.pe_type == "laplacian":
        transform.append(T.AddLaplacianEigenvectorPE(k=cfg.dataset.transform.pe_size, attr_name='pe'))
    elif cfg.dataset.transform.pe_type == "random_walk":
        transform.append(T.AddRandomWalkPE(cfg.dataset.transform.pe_size, attr_name='pe'))
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

    elif cfg.dataset.id == "bitcoin":

        train_dataset = BitcoinOTC(root=cfg.dataset.root, transform=transform)
        # in this case node masks are used to split the dataset
        test_dataset = train_dataset
        use_node_mask = cfg.dataset.bitcoin.node_mask

    # Define the model 
    model = SignDenoising(16, node_attr_size)
    model2 = SignDenoising2(16, node_attr_size)
    training = Training(
        cfg=cfg,
        model=model2)

    # Train and test
    training.train(dataset=train_dataset, epochs=10, use_node_mask=use_node_mask)
    training.test(dataset=test_dataset, use_node_mask=use_node_mask)

if __name__ == "__main__":
    main()