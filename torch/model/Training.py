from model import SpringTransform, log_regression
from timeit import default_timer as timer
import torch

class Training:
    """
    Training
    """
    def __init__(self, 
        device, 
        train_data, 
        test_data, 
        embedding_dim, 
        time_step, 
        iterations, 
        damping,
        friend_distance,
        friend_stiffness):

        self.device = device
        self.train_data = train_data
        self.test_data = test_data
        self.embedding_dim = embedding_dim
        self.time_step = time_step
        self.iterations = iterations
        self.damping = damping
        self.friend_distance = friend_distance
        self.friend_stiffness = friend_stiffness

        self.ratio = torch.count_nonzero(self.train_data.edge_attr ==1) / torch.count_nonzero(self.train_data.edge_attr == -1)
        print(f"Ratio: {self.ratio}")

    def __call__(self, 
        neutral_distance, 
        neutral_stiffness, 
        enemy_distance,     
        enemy_stiffness) -> float:

        transform = SpringTransform(
            device=self.device,
            embedding_dim=self.embedding_dim,
            time_step=self.time_step,
            iterations=self.iterations,
            damping=self.damping,
            friend_distance=self.friend_distance,
            friend_stiffness=self.friend_stiffness,
            neutral_distance=neutral_distance,
            neutral_stiffness=neutral_stiffness,
            enemy_distance=enemy_distance,
            enemy_stiffness=enemy_stiffness,
        )

        start = timer()
        self.train_data = transform(self.train_data)
        end = timer()
        print(f"Time: {end - start}")
        
        auc, f1_binary, f1_micro, f1_macro, y_pred =\
            log_regression(self.train_data, self.test_data)
        
        self.y_pred = y_pred

        print(f"AUC: {auc}")
        print(f"F1 Binary: {f1_binary}")
        print(f"F1 Micro: {f1_micro}")
        print(f"F1 Macro: {f1_macro}")

        return 1 / f1_macro