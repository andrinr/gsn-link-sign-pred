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
    """
    Biased multi headed self attention layer
    """
    def __init__(
        self,
        vector_size : int,
        number_of_heads : int = 8
    ):
        self.number_of_heads = number_of_heads

        # The linear layers are identical to matrix multiplication with weight matrix that is learned
        self.queries = nn.Linear(vector_size, vector_size * number_of_heads)
        self.keys = nn.Linear(vector_size, vector_size * number_of_heads)
        self.values = nn.Linear(vector_size, vector_size * number_of_heads)
    
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

        queries_nodes = self.queries(node_features)
        keys_nodes = self.keys(node_features)

        queries_edges = self.queries(edge_features)
        keys_edges = self.keys(edge_features)

    def attention(
        query : torch.Tensor,
        key : torch.Tensor,
        value : torch.Tensor,
        mask : torch.Tensor = None,
        dropout : torch.nn.Dropout = None
    ):
        """
        Computes scaled dot product attention
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn






class GraphTransformer(nn.Module):
    def __init__(self, )
        super().__init__()


