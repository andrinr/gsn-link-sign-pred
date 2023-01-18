import torch
from data.diffusion import node_sign_diffusion
from model.denoising import SignDenoising
import numpy as np

class Training:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model

    def train(self, dataset, epochs=20):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            for data in dataset:
                predictions, truth = self.step(data)
                loss = criterion(predictions, truth)
                #print(sign_predictions, d_true_signs)
                print(loss.item())
                loss.backward()
                optimizer.step()


    def test(self, dataset):
        with torch.no_grad():
            for data in dataset:
                predictions, truth = self.step(data)
                loss = self.criterion(predictions, truth)

                print(loss.item())


    def step(self, data):
        true_signs = data.x
        diffused_signs = node_sign_diffusion(true_signs, np.random.random())
        # squeeze
        true_signs = torch.squeeze(true_signs)
        pe = data['pe']
        # generate x
        x = torch.cat([diffused_signs, pe, ], dim=1)
        # squeeze data 
        true_signs = torch.squeeze(true_signs)
        # send data to device
        d_x = x.to(self.device)
        d_edge_index = data.edge_index.to(self.device)
        d_true_signs = true_signs.to(self.device)
        # make prediction
        return self.model(d_x, d_edge_index), d_true_signs
        