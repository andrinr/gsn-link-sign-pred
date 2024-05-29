from simulation import SpringState
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import jax.numpy as jnp
from typing import NamedTuple

class Metrics(NamedTuple):
    auc : float
    f1_binary : float
    f1_micro : float
    f1_macro : float
    true_positives : float
    false_positives : float
    true_negatives : float
    false_negatives : float

    def __str__(self):
        return "auc: {self.auc}," + \
        "f1_binary: {self.f1_binary}, " + \
        "f1_micro: {self.f1_micro}, " + \
        "f1_macro: {self.f1_macro}, " + \
        "true_positives: {self.true_positives}, " + \
        "false_positives: {self.false_positives}, " + \
        "true_negatives: {self.true_negatives}, " + \
        "false_negatives: {self.false_negatives}"

def evaluate(
    spring_state : SpringState, 
    edge_index : jnp.ndarray,
    signs : jnp.ndarray,
    training_mask : jnp.ndarray,
    evaulation_mask : jnp.ndarray) -> Metrics:

    logreg = LogisticRegression()
        
    embeddings = spring_state.position
    position_i = embeddings.at[edge_index[0]].get()
    position_j = embeddings.at[edge_index[1]].get()

    spring_vec_norm = jnp.linalg.norm(position_i - position_j, axis=1)
    spring_vec_norm = jnp.expand_dims(spring_vec_norm, axis=1)

    logreg.fit(spring_vec_norm.at[training_mask].get(), signs.at[training_mask].get())
    y_pred = logreg.predict(spring_vec_norm.at[evaulation_mask].get())

    try:
        auc = roc_auc_score(signs.at[evaulation_mask].get(), y_pred)
        f1_binary = f1_score(signs.at[evaulation_mask].get(), y_pred, average='binary')
        f1_micro = f1_score(signs.at[evaulation_mask].get(), y_pred, average='micro')
        f1_macro = f1_score(signs.at[evaulation_mask].get(), y_pred, average='macro')
        tn, fp, fn, tp = confusion_matrix(signs.at[evaulation_mask].get(), y_pred).ravel()

    except ValueError:
        auc = 0
        f1_binary = 0
        f1_micro = 0
        f1_macro = 0
        tn, fp, fn, tp = 0, 0, 0, 0

    return Metrics(auc, f1_binary, f1_micro, f1_macro, tp, fp, tn, fn), y_pred