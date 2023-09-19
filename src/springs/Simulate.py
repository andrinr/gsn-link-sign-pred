from springs import SpringTransform, log_regression
from timeit import default_timer as timer
import torch
import matplotlib.pyplot as plt

class Training:
    """
    Training
    """
    def __init__(self, 
        train_data, 
        test_data, 
        embedding_dim, 
        time_step, 
        iterations, 
        damping,
        friend_distance,
        friend_stiffness):

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

        iterations_interval = 100
        num_intervals = self.iterations // iterations_interval

        transform = SpringTransform(
            embedding_dim=self.embedding_dim,
            time_step=self.time_step,
            iterations=iterations_interval,
            damping=self.damping,
            friend_distance=self.friend_distance,
            friend_stiffness=self.friend_stiffness,
            neutral_distance=neutral_distance,
            neutral_stiffness=neutral_stiffness,
            enemy_distance=enemy_distance,
            enemy_stiffness=enemy_stiffness,
        )

        start = timer()

        aucs = []
        f1_binaries = []
        f1_micros = []
        f1_macros = []
        energies = []

        for i in range(num_intervals):
            self.train_data = transform(self.train_data)

            auc, f1_binary, f1_micro, f1_macro, y_pred =\
                log_regression(self.train_data, self.test_data)
            
            aucs.append(auc)
            f1_binaries.append(f1_binary)
            f1_micros.append(f1_micro)
            f1_macros.append(f1_macro)
            energies.append(transform.energy_total)

        end = timer()
        print(f"Time: {end - start}")

        self.energies = transform.energies

        self.y_pred = y_pred

        # plot measures and energy
        plt.plot(aucs, label='AUC')
        plt.plot(f1_binaries, label='F1 binary')
        plt.plot(f1_micros, label='F1 micro')
        plt.plot(f1_macros, label='F1 macro')
        plt.plot(energies, label='Energy')

        plt.legend()
        plt.show()

        return 1 / f1_macro