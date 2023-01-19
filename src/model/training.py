import torch
from data import node_sign_diffusion
import numpy as np

class Training:
    def __init__(self, cfg, model, offset_unbalanced=False):
        self.cfg = cfg
        self.model = model
        self.offset_unbalanced = offset_unbalanced

    def train(self, dataset, epochs=20, use_node_mask=False):

        print(dataset[0].x.shape)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        if self.offset_unbalanced:
            weights = torch.Tensor(np.array([
                self.cfg.dataset.p_positive, 
                1-self.cfg.dataset.p_positive]))

            d_weights = weights.to(self.device)

            criterion = torch.nn.CrossEntropyLoss(weight=d_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            for data in dataset:
                optimizer.zero_grad()
                time = np.random.random() * np.random.random()
                predictions, target = self.step(data, time)
                predicted_class = torch.argmax(predictions, 1)
                loss = criterion(predictions, target)
                #print(sign_predictions, d_true_signs)
                print(loss.item())
                loss.backward()
                optimizer.step()

    def test(self, dataset, use_node_mask=False):
        with torch.no_grad():
            acc = []
            for data in dataset:
                predictions, target = self.step(data, 0.95)
                predicted_class = torch.argmax(predictions, 1)
                print(predicted_class)
                correct = (predicted_class == target).float()
                acc.append(correct.sum() / len(correct))

            print(f"Test accuracy: {sum(acc) / len(acc)}")

    def step(self, data, diffusion_time):
        target = data.x
        diffused = node_sign_diffusion(target, diffusion_time)
        pe = data['pe']
        x = torch.cat([diffused, pe], dim=1)
        target = torch.squeeze(target)

        #print(diffused, target )
        #print('noisage', (target == torch.squeeze(diffused)).sum().item() / len(target))

        # send data to device
        d_x = x.to(self.device)
        d_edge_index = data.edge_index.to(self.device)
        d_target = target.to(self.device)

        # make prediction
        return self.model(d_x, d_edge_index), d_target
        