# External dependencies
import numpy as np
import hydra
from omegaconf import DictConfig
import networkx as nx 
import matplotlib.pyplot as plt
from functools import partial
# torch imports
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
# Local dependencies
from data.BSCLGraph import BSCLGraph
from data.SignedDataset import SignedDataset
from data.utils.samplers import even_exponential
from data.diffusion import node_sign_diffusion
from model.denoising import SignDenoising

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
    # Signs are not node features and note edge features anymore
    node_attr_size = cfg.dataset.laplacian_eigenvector_pe_size + 1
    transform = T.Compose([
        T.LineGraph(force_directed=True),
        T.AddLaplacianEigenvectorPE(k=cfg.dataset.laplacian_eigenvector_pe_size, attr_name='pe')])

    dataset = SignedDataset(
        graph_generator=BSCLGraph,
        graph_generator_kwargs=BSCL_graph_kwargs,
        transform=transform,
        num_graphs=cfg.dataset.n_graphs)

    print('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignDenoising(16, node_attr_size).to(device)

    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    for epoch in range(20):
        print(f"Epoch {epoch}")
        for data in dataset:
            true_signs = data.x
            diffused_signs = node_sign_diffusion(true_signs, np.random.random())
            # squeeze
            true_signs = torch.squeeze(true_signs)
            pe = data['pe']
            # generate x
            x = torch.cat([diffused_signs, pe], dim=1)
            # squeeze data 
            true_signs = torch.squeeze(true_signs)
            # send data to device
            d_x = x.to(device)
            d_edge_index = data.edge_index.to(device)
            d_true_signs = true_signs.to(device)
            # make prediction
            sign_predictions = model(d_x, d_edge_index)
            loss = criterion(sign_predictions, d_true_signs)
            #print(sign_predictions, d_true_signs)
            print(loss.item())
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()