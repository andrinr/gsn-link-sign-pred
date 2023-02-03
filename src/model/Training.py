from model import SpringTransform, log_regression

class Training:
    def __init__(self, 
        device, 
        train_data, 
        test_data, 
        test_mask, 
        embedding_dim, 
        time_step, 
        iterations, 
        damping):

        self.device = device
        self.train_data = train_data
        self.test_data = test_data
        self.test_mask = test_mask
        self.embedding_dim = embedding_dim
        self.time_step = time_step
        self.iterations = iterations
        self.damping = damping

    def __call__(self, 
        friend_distance, 
        friend_stiffness, 
        neutral_distance, 
        neutral_stiffness, 
        enemy_distance, 
        enemy_stiffness) -> float:

        print(f"friend_distance: {friend_distance}, friend_stiffness: {friend_stiffness}, neutral_distance: {neutral_distance}, neutral_stiffness: {neutral_stiffness}, enemy_distance: {enemy_distance}, enemy_stiffness: {enemy_stiffness}")
        transform = SpringTransform(
            device=self.device,
            embedding_dim=self.embedding_dim,
            time_step=self.time_step,
            iterations=self.iterations,
            damping=self.damping,
            friend_distance=friend_distance,
            friend_stiffness=friend_stiffness,
            neutral_distance=neutral_distance,
            neutral_stiffness=neutral_stiffness,
            enemy_distance=enemy_distance,
            enemy_stiffness=enemy_stiffness,
        )

        self.train_data = transform(self.train_data)

        auc, acc, prec, rec, f1 = log_regression(self.train_data, self.test_data, self.test_mask)

        print(f"AUC: {auc}")
        print(f"Accuracy: {acc}")
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1: {f1}")

        return 1 / auc