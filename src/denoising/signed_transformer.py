import torch
import torch.nn as nn

class Feature():
    def __init__(
        self,
        feature_size : int,
        hidden_size : int
    )
        self.feature_size = feature_size
        self.hidden_size = hidden_size

class SignedGraphTransformer(nn.Module):

    def __init__(
        self,
        node_features : Feature,
        edge_features : Feature,
        embeddings : Feature,

    ):
        self.activation_function == nn.ReLU()

        self.linear_transformation_nodes = nn.Linear(
            node_features.feature_size, 
            node_features.hidden_size
            self.activation_function)

        self.linear_transformation_nodes = nn.Linear(
            edge_features.feature_size,
            edge_features.hidden_size
            self.activation_function)

        self.linear_transformation_embeddings = nn.Linear(
            embeddings.feature_size,
            embeddings.hidden_size
            self.activation_function)

        
    def forward(self, X, E, y):
        hidden_node_features = self.linear_transformation_nodes(X)
        hidden_edge_features = self.linear_transformation_edges(E)
        hidden_embedding_features = self.linear_transformation_embeddings(y)

        hidden_node_features += hidden_embedding_features

        

        



