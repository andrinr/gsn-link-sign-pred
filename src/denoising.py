import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import GraphTransformer

class DenoisingModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_dtype = torch.float32
        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                    input_dims=input_dims,
                                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                    hidden_dims=cfg.model.hidden_dims,
                                    output_dims=output_dims,
                                    act_fn_in=nn.ReLU(),
                                    act_fn_out=nn.ReLU())

        

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)