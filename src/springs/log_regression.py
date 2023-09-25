import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

def log_regression(train_data, test_data):
    test_mask = test_data.edge_attr != 0
    n_train = torch.count_nonzero(test_mask == 0).item()
    n_test = torch.count_nonzero(test_mask == 1).item()
    n_total = n_train + n_test

    X_train = torch.zeros(n_train)
    y_train = torch.zeros(n_train)
    X_test = torch.zeros(n_test)
    y_test = torch.zeros(n_test)

    y_pred_sparse = torch.zeros(n_total)

    l = 0
    indices = torch.arange(n_total)
    indices = indices[train_data.edge_attr != 0]
    pbar = tqdm(indices)
    pbar.set_description("Prepare log regression training data")

    for k in pbar:
        i = train_data.edge_index[0, k]
        j = train_data.edge_index[1, k]
        pos_i = train_data.x[i]
        pos_j = train_data.x[j]
        X_train[l] = torch.dist(pos_i, pos_j)
        y_train[l] = train_data.edge_attr[k]
        l += 1

    X_train = torch.unsqueeze(X_train, 1)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    indices = torch.arange(n_total)
    indices = indices[test_data.edge_attr != 0]
    pbar = tqdm(indices)
    pbar.set_description("Prepare log regression test data")
    l = 0
    for k in pbar:
        i = test_data.edge_index[0, k]
        j = test_data.edge_index[1, k]
        # The node features are not available in the test data
        pos_i = train_data.x[i]
        pos_j = train_data.x[j]
        X_test[l] = torch.dist(pos_i, pos_j)
        y_test[l] = test_data.edge_attr[k]
        l += 1

    X_test = torch.unsqueeze(X_test, 1)
    y_pred = clf.predict(X_test)
    y_pred_sparse[test_mask] = torch.tensor(y_pred)
    y_pred = y_pred / 2 + 0.5
    y_test = y_test / 2 + 0.5
    y_pred_prob = clf.predict_proba(X_test)

    # evaluate the performance of the classifier
    auc_score = roc_auc_score(y_test, y_pred_prob[:, 1])

    f1_binary = f1_score(y_test, y_pred, average='binary', pos_label=1)
    f1_micro = f1_score(y_test, y_pred, average='micro', pos_label=1)
    f1_macro = f1_score(y_test, y_pred, average='macro', pos_label=1)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print('True negatives: ', tn, '\nFalse positives: ', fp, '\nFalse negatives: ', fn, '\nTrue Positives: ', tp)

    return auc_score, f1_binary, f1_micro, f1_macro, y_pred_sparse