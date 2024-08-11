from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import jax.numpy as jnp
import simulation as sm

def evaluate(
    node_state : sm.NodeState, 
    edge_index : jnp.ndarray,
    signs : jnp.ndarray,
    training_mask : jnp.ndarray,
    evaulation_mask : jnp.ndarray) -> sm.Metrics:

    logreg = LogisticRegression()
        
    embeddings = node_state.position
    position_i = embeddings.at[edge_index[0]].get()
    position_j = embeddings.at[edge_index[1]].get()

    spring_vec_norm = jnp.linalg.norm(position_i - position_j, axis=1)
    spring_vec_norm = jnp.expand_dims(spring_vec_norm, axis=1)

    logreg.fit(spring_vec_norm.at[training_mask].get(), signs.at[training_mask].get())
    y_pred = logreg.predict(spring_vec_norm.at[evaulation_mask].get())

    # get y_pred probabilities
    y_pred_prob = logreg.predict_proba(spring_vec_norm.at[evaulation_mask].get())[:, 1]

    try:
        auc_l = roc_auc_score(signs.at[evaulation_mask].get(), y_pred)
        auc_p = roc_auc_score(signs.at[evaulation_mask].get(), y_pred_prob)
        f1_binary = f1_score(signs.at[evaulation_mask].get(), y_pred, average='binary')
        f1_micro = f1_score(signs.at[evaulation_mask].get(), y_pred, average='micro')
        f1_macro = f1_score(signs.at[evaulation_mask].get(), y_pred, average='macro')
        f1_weighted = f1_score(signs.at[evaulation_mask].get(), y_pred, average='weighted')
        
        tn, fp, fn, tp = confusion_matrix(signs.at[evaulation_mask].get(), y_pred).ravel()

    except ValueError:
        auc_l = 0
        auc_p = 0
        f1_binary = 0
        f1_micro = 0
        f1_macro = 0
        f1_weighted = 0
        tn, fp, fn, tp = 0, 0, 0, 0

    return sm.Metrics(
        auc_l = auc_l,
        auc_p = auc_p,
        f1_binary = f1_binary,
        f1_micro = f1_micro,
        f1_macro = f1_macro,
        f1_weighted = f1_weighted,
        true_positives = tp,
        false_positives = fp,
        true_negatives = tn,
        false_negatives = fn), y_pred