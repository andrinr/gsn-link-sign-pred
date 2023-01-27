import torch
import torch.nn.functional as F
from data import node_sign_diffusion
import numpy as np
from model import generate_embedding

class Training:
    def __init__(self, cfg, model, device):
        self.cfg = cfg
        self.model = model
<<<<<<< HEAD
        self.device = device

    def train(self, dataset, epochs=20):
=======

    def train(self, dataset, epochs=20):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for data in dataset:
            z = generate_embedding(self.device, self.cfg.embedding, get_edge_index(data))
            data.x = torch.cat([data.x, z], dim=1)

>>>>>>> 575938ec92c353b1620effe935986de8c2808464
        self.model.to(self.device)

        if self.cfg.balance_loss:
            negative_fraction = (torch.count_nonzero(dataset[0].x[:,0] == 1) / dataset[0].num_nodes).item()
            positive_fraction = 1.0 - negative_fraction
            weights = torch.Tensor(np.array([
                positive_fraction / negative_fraction, 
                1]))
<<<<<<< HEAD
            print(weights)
=======
>>>>>>> 575938ec92c353b1620effe935986de8c2808464
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
<<<<<<< HEAD
                predictions, target, _ = self.step(data, 0.1)
                target_class = torch.argmax(target, 1)
                loss = criterion(predictions, target_class)
                print(predictions, target)
=======
                predictions, target = self.step(data, 0.95)
                target_class = torch.argmax(target, 1)
                loss = criterion(predictions, target_class)
                #print(sign_predictions, d_true_signs)
>>>>>>> 575938ec92c353b1620effe935986de8c2808464
                print('loss', loss.item())
                loss.backward()
                optimizer.step()

    def test(self, dataset):
        with torch.no_grad():
            acc = []
            for data in dataset:
<<<<<<< HEAD
                predictions, target, mask = self.step(data, 0.1)
                target_class = torch.argmax(target, 1)
                predicted_class = torch.argmax(predictions, 1)
                
=======
                predictions, target = self.step(data, 0.75)
                target_class = torch.argmax(target, 1)
                predicted_class = torch.argmax(predictions, 1)
                
                print(predicted_class)
>>>>>>> 575938ec92c353b1620effe935986de8c2808464
                correct = (predicted_class == target_class).float()
                print('correct new', correct[mask].sum(), len(correct[mask]))
                print('correct old', correct[~mask].sum(), len(correct[~mask]))
                acc.append(correct.sum() / len(correct))

            print(f"Test accuracy: {sum(acc) / len(acc)}")

    def step(self, data, diffusion_time):
        target = data.x[:, :2]
        attibutes = data.x[:, 2:]

<<<<<<< HEAD
        diffused, mask = node_sign_diffusion(target, diffusion_time)
=======
        diffused = node_sign_diffusion(target, diffusion_time)
>>>>>>> 575938ec92c353b1620effe935986de8c2808464
        x = torch.cat([diffused, attibutes], dim=1)
        #print(diffused, target )
        print('noisage', mask.sum() / len(x))


        # send data to device
        d_x = x.to(self.device)
        # Either select sparse or dense edge index format
        d_edge_index = get_edge_index(data).to(self.device)
        d_target = target.to(self.device)

        # make prediction
<<<<<<< HEAD
        return self.model(d_x, d_edge_index), d_target, mask
=======
        return self.model(d_x, d_edge_index), d_target
>>>>>>> 575938ec92c353b1620effe935986de8c2808464

def get_edge_index(data):
    if not data.edge_index:
        return data.adj_t
    else:
        return data.edge_index
        