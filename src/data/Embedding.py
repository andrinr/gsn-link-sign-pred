import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn.models import Node2Vec

class Embedding(BaseTransform):
    def __init__(
        self,
        device,
        cat : bool,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        walks_per_node: int,
        epochs: int,
    ):
        self.device = device
        self.cat = cat
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.epochs = epochs

    def __call__(self, data: Data) -> Data:
        node2vec = Node2Vec(
            data.edge_index, 
            embedding_dim=self.embedding_dim,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            num_negative_samples=1, p=1, q=1, sparse=True).to(self.device)

        loader = node2vec.loader(batch_size=128, shuffle=True,
                            num_workers=0)

        optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

        def train():
            node2vec.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = node2vec.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)

        for epoch in range(1, self.epochs + 1):
            print(epoch)
            loss = train()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


        node2vec.eval()
        z = node2vec(torch.arange(data.num_nodes, device=self.device))

        x = data.x

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, z.cpu().to(x.dtype)], dim=-1)
        else:
            data.x = z.cpu()

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.cat})'