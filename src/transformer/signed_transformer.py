import torch
import torch.nn as nn

from utils import Feature

class SignedGraphTransformer(nn.Module):

    def __init__(
        self,
        node_features : Feature,
        edge_features : Feature,
        embeddings : Feature
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

        
    def forward(
        self, 
        node_features, 
        edge_features, 
        node_embeddings
    ):
        hidden_node_features = self.linear_transformation_nodes(node_features)
        hidden_edge_features = self.linear_transformation_edges(edge_features)
        hidden_embedding_features = self.linear_transformation_embeddings(node_embeddings)

        hidden_node_features += hidden_embedding_features


class SelfAttention(nn.Module):
    def __init__(
        self,
        vector_size : int,
        number_of_heads : int = 8
    ):
        self.number_of_heads = number_of_heads

        self.queries = nn.Linear(vector_size, vector_size)
        self.keys = nn.Linear(vector_size, vector_size)
        self.values = nn.Linear(vector_size, vector_size)
    
    def forward(
        self,
        node_features : torch.Tensor,
        edge_features : torch.Tensor,
        node_embeddings : torch.Tensor,
        mask : torch.Tensor
    ) -> torch.Tensor, torch.Tensor, torch.Tensor:
        node_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)

        Q = self.queries(node_features)
        K = self.keys(node_features)





class GraphTransformer(nn.Module):
    def __init__(self, )
        super().__init__()


