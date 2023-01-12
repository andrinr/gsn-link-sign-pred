import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import GraphTransformer
from diffusion import graph_diffusion

class DenoisingModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_dtype = torch.float32

        input_dims = {
            'X' : 0,
            'E' : 1,
            'y' : 0
        }

        output_dims = {
            'X' : 0,
            'E' : 1,
            'y' : 0
        }

        self.model = GraphTransformer(n_layers=cfg.n_layers,
                                    input_dims=input_dims,
                                    hidden_mlp_dims=cfg.hidden_mlp_dims,
                                    hidden_dims=cfg.hidden_dims,
                                    output_dims=output_dims,
                                    act_fn_in=nn.ReLU(),
                                    act_fn_out=nn.ReLU())

    def training_step(self, batch, batch_idx):
        pass
        random_graph = 0
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def forward(self, X, E, y, node_mask):
        return self.model(X, E, y, node_mask)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)