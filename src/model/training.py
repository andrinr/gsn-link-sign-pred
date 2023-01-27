import torch
import torch.nn.functional as F
from data import node_sign_diffusion
import numpy as np

class Training:
    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.model = model
        self.device = device

    def train(self, dataset, epochs=20):
        self.model.to(self.device)

        if self.cfg.balance_loss:
            negative_fraction = (torch.count_nonzero(dataset[0].x[:,0] == 1) / dataset[0].num_nodes).item()
            positive_fraction = 1.0 - negative_fraction
            weights = torch.Tensor(np.array([
                positive_fraction / negative_fraction, 
                1]))
            print(weights)
            d_weights = weights.to(self.device)

            criterion = torch.nn.CrossEntropyLoss(weight=d_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        print("Training")
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            for data in dataset:
                optimizer.zero_grad()
                time = np.random.random() * np.random.random()
                predictions, target, _ = self.step(data, 0.1)
                target_class = torch.argmax(target, 1)
                loss = criterion(predictions, target_class)
                print(predictions, target)
                print('loss', loss.item())
                loss.backward()
                optimizer.step()

    def test(self, dataset):
        with torch.no_grad():
            acc = []
            for data in dataset:
                predictions, target, mask = self.step(data, 0.1)
                target_class = torch.argmax(target, 1)
                predicted_class = torch.argmax(predictions, 1)
                
                correct = (predicted_class == target_class).float()
                print('correct new', correct[mask].sum(), len(correct[mask]))
                print('correct old', correct[~mask].sum(), len(correct[~mask]))
                acc.append(correct.sum() / len(correct))

            print(f"Test accuracy: {sum(acc) / len(acc)}")

    def step(self, data, diffusion_time):
        target = data.x[:, :2]
        attibutes = data.x[:, 2:]

        diffused, mask = node_sign_diffusion(target, diffusion_time)
        x = torch.cat([diffused, attibutes], dim=1)
        #print(diffused, target )
        print('noisage', mask.sum() / len(x))


        # send data to device
        d_x = x.to(self.device)
        # Either select sparse or dense edge index format
        d_edge_index = get_edge_index(data).to(self.device)
        d_target = target.to(self.device)

        # make prediction
        return self.model(d_x, d_edge_index), d_target, mask

def get_edge_index(data):
    if not data.edge_index:
        return data.adj_t
    else:
        return data.edge_index
        