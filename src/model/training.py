import torch
import torch.nn.functional as F
from data import node_sign_diffusion
import numpy as np

class Training:
    def __init__(self, cfg, model, offset_unbalanced=False):
        self.cfg = cfg
        self.model = model
        self.offset_unbalanced = offset_unbalanced

    def train(self, dataset, epochs=20):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        if self.offset_unbalanced:
            negative_fraction = (torch.count_nonzero(dataset[0].x == 0) / dataset[0].num_nodes).item()
            positive_fraction = 1.0 - negative_fraction
            weights = torch.Tensor(np.array([
                positive_fraction, 
                negative_fraction]))

            print(negative_fraction, positive_fraction)

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
                predictions, target = self.step(data, time)
                predicted_class = torch.argmax(predictions, 1)
                loss = criterion(predictions, target)
                #print(sign_predictions, d_true_signs)
                print('loss', loss.item())
                loss.backward()
                optimizer.step()

    def test(self, dataset):
        with torch.no_grad():
            acc = []
            for data in dataset:
                predictions, target = self.step(data, 0.75)
                target_class = torch.argmax(target, 1)
                predicted_class = torch.argmax(predictions, 1)
                
                print(predicted_class)
                correct = (predicted_class == target_class).float()
                acc.append(correct.sum() / len(correct))

            print(f"Test accuracy: {sum(acc) / len(acc)}")

    def step(self, data, diffusion_time):
        target = data.x
        
        diffused = node_sign_diffusion(target, diffusion_time)
        diffused = F.one_hot(diffused, num_classes=2).float()
        #diffused = torch.squeeze(diffused)
        target = F.one_hot(target, num_classes=2).float()
        target = torch.squeeze(target)

        #pe = data['pe']
        index = np.linspace(0, len(target)-1, len(target))
        np.random.shuffle(index)
        index = torch.tensor(index, dtype=torch.long)
        index = torch.unsqueeze(index, dim=-1)

        x = torch.cat([diffused, index], dim=1)
        #print(diffused, target )
        #print('noisage', (target == torch.squeeze(diffused)).sum().item() / len(target))

        # send data to device
        d_x = x.to(self.device)
        # Either select sparse or dense edge index format
        if not data.edge_index:
            d_edge_index = data.adj_t.to(self.device)
        else:
            d_edge_index = data.edge_index.to(self.device)
        d_target = target.to(self.device)

        # make prediction
        return self.model(d_x, d_edge_index), d_target
        