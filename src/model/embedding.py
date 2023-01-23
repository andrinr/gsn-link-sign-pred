import torch
from model import Node2Vec

def generate_embedding(device, cfg, edge_index):
    print("Generating embedding")
    print(edge_index)
    node2vec = Node2Vec(
        edge_index, 
        embedding_dim=cfg.embedding_dim, 
        walk_length=cfg.walk_length,
        context_size=cfg.context_size,
        walks_per_node=cfg.walks_per_node,
        num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = node2vec.loader(batch_size=128, shuffle=True,
                        num_workers=0)

    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

    def train():
        node2vec.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(1, cfg.epochs + 1):
        print(epoch)
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    return node2vec(edge_index)