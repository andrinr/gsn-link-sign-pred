import jax.numpy as jnp
from jax import jit

from sklearn.linear_model import LogisticRegression

class LogReg:

    def __init__(self,
        pos_i : jnp.ndarray,
        pos_j : jnp.ndarray,
        mask : jnp.ndarray,
        actual_sign : jnp.ndarray):

        self.pos_i = pos_i
        self.pos_j = pos_j
        self.mask = mask
        self.actual_sign = actual_sign

        self.scale = 0.1
        self.location = 0.0

        self.clf = LogisticRegression() 

    def fit(self):

        X = jnp.linalg.norm(self.pos_i - self.pos_j, axis=1, keepdims=True)
        X = self.log_f(X)
        X = X[self.mask]
        y = self.actual_sign[self.mask]

        X = jnp.expand_dims(X, axis=1)
        y = jnp.expand_dims(y, axis=1)

        self.clf.fit(X, y)

        
    def predict(self, pos_i, pos_j):
        X = jnp.linalg.norm(pos_i - pos_j, axis=1, keepdims=True)
        
        y = self.clf.predict(X)

        auc_score = roc_auc_score(y_test, y_pred_prob[:, 1])

        f1_binary = f1_score(y_test, y_pred, average='binary', pos_label=1)
        f1_micro = f1_score(y_test, y_pred, average='micro', pos_label=1)
        f1_macro = f1_score(y_test, y_pred, average='macro', pos_label=1)

        return y

        

