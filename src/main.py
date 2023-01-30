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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

        # iterate over each edge in the pyg graph
    """for i, j in train_data.edge_index.t().tolist():
        distance = torch.norm(train_data.x[i] - train_data.x[j])
        edge_indices = torch.where((train_data.edge_index[0] == i) & (train_data.edge_index[1] == j))
        actual_sign = train_data.edge_attr[edge_indices]
        print(f"Edge {i} -> {j} has distance {distance} and sign {actual_sign}")'"""

    X_train = torch.ones((len(train_data.edge_index[0]), 2 * train_data.x.shape[1]))
    y_train = torch.ones((len(train_data.edge_index[0]), 1))
    edge_list = train_data.edge_index.t().tolist()
    for k in range(len(edge_list)):
        i, j = edge_list[k]
        pos_i = train_data.x[i]
        pos_j = train_data.x[j]
        X_train[k] = torch.cat((pos_i, pos_j))
        y_train[k] = train_data.edge_attr[
            torch.where((train_data.edge_index[0] == i) & (train_data.edge_index[1] == j))]

    clf = LogisticRegression()
    y_train = torch.squeeze(y_train)
    print(X_train)
    print(y_train)
    clf.fit(X_train, y_train)

    X_test = torch.ones((len(train_data.edge_index[0]), 2 * train_data.x.shape[1]))
    y_test = torch.ones((len(train_data.edge_index[0]), 1))
    edge_list = train_data.edge_index.t().tolist()
    for k in range(len(edge_list)):
        i, j = edge_list[k]
        pos_i = train_data.x[i]
        pos_j = train_data.x[j]
        X_test[k] = torch.cat((pos_i, pos_j))
        y_test[k] = test_data.edge_attr[torch.where((test_data.edge_index[0] == i) & (test_data.edge_index[1] == j))]
    
    y_test = torch.squeeze(y_test)
    y_pred = clf.predict(X_test)

    # evaluate the performance of the classifier
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    """peModel = DGCNN(emb_size=64)

    training_pe = Training(
        cfg=cfg,
        model=peModel,
        offset_unbalanced=False
    )"""




if __name__ == "__main__":
    main()