# External dependencies
import hydra
from omegaconf import DictConfig
from functools import partial
# torch imports
import torch_geometric.transforms as T
# Local dependencies
from data.BSCLGraph import BSCLGraph
from data.SignedDataset import SignedDataset
from data.utils.samplers import even_exponential
from model.denoising import SignDenoising
from model.training import Training

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:

    degree_generator = partial(even_exponential, size=cfg.dataset.n_nodes, scale=5.0)
    BSCL_graph_kwargs = {
        "degree_generator": degree_generator,
        "p_positive_sign": cfg.dataset.BSCL.p_positive,
        "p_close_triangle": cfg.dataset.BSCL.p_close_triangle,
        "p_close_for_balance": cfg.dataset.BSCL.p_close_for_balance,
        "remove_self_loops": cfg.dataset.BSCL.remove_self_loops
    }

    # We transform the graph to a line graph, meaning each edge is replaced with a node
    # Source: Line Graph Neural Networks for Link Prediction
    # Signs are now node features
    node_attr_size = cfg.dataset.pe_size + 1
    transform = T.Compose([
        T.LineGraph(force_directed=True),
        #T.AddLaplacianEigenvectorPE(k=cfg.dataset.pe_size, attr_name='pe')
        T.AddRandomWalkPE(cfg.dataset.pe_size, attr_name='pe')
        ])

    model = SignDenoising(16, node_attr_size)

    print("Loading dataset")
    train_dataset = SignedDataset(
        graph_generator=BSCLGraph,
        graph_generator_kwargs=BSCL_graph_kwargs,
        transform=transform,
        num_graphs=cfg.dataset.train_size)

    test_dataset = SignedDataset(
        graph_generator=BSCLGraph,
        graph_generator_kwargs=BSCL_graph_kwargs,
        transform=transform,
        num_graphs=cfg.dataset.test_size)
    
    training = Training(
        cfg=cfg,
        model=model)

    training.train(dataset=train_dataset, epochs=20)

    training.test(dataset=test_dataset)
if __name__ == "__main__":
    main()