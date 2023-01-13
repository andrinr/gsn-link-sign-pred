import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import GraphTransformer

class SignDenoisingPipeline(pl.LightningModule):
    def __init__(self, dataset, cfg):
        super(SignDenoising, self).__init__()
        self.cfg = cfg
        self.model_dtype = torch.float32
        self.dataset = dataset

        self.nn_layers = []
        self.nn_layers.append(nn.Linear(1, 1))
        self.nn_layers.append(nn.Linear(1, 1))


    def training_step(self, batch, batch_idx):
        print(batch)
  
        return 0
        

    def forward(self, x, node_mask):
        
        for layer in self.nn_layers:
            x = layer(x)
            x = F.relu(x)
        
        return x
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)