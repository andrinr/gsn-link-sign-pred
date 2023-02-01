import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def log_regression(train_data, test_data, test_mask):

    n_edges = test_data.edge_index.shape[1]
    print(f"Number of nodes: {test_data.num_nodes}")
    print(f"Number of positive edges: {test_data.edge_attr.sum()}")
    print(f"Number of negative edges: {n_edges - test_data.edge_attr.sum()}")
    print(f"Ratio of positive edges: {test_data.edge_attr.sum() / n_edges}")

    print(test_mask)
    n_train = torch.count_nonzero(test_mask == 0).item()
    n_test = torch.count_nonzero(test_mask == 1).item()
    n_total = n_train + n_test

    X_train = torch.zeros(n_train)
    y_train = torch.zeros(n_train)
    X_test = torch.zeros(n_test)
    y_test = torch.zeros(n_test)

    l = 0
    for k in range(n_total):
        if train_data.edge_attr[k] == 0:
            continue
        i = train_data.edge_index[0, k]
        j = train_data.edge_index[1, k]
        pos_i = train_data.x[i]
        pos_j = train_data.x[j]
        X_train[l] = torch.dist(pos_i, pos_j)
        y_train[l] = train_data.edge_attr[k]
        l += 1

    X_train = torch.unsqueeze(X_train, 1)
    print(0, torch.count_nonzero(y_train==0), n_train)
    print(1, torch.count_nonzero(y_train==1), n_train)
    print(-1, torch.count_nonzero(y_train==-1), n_train)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    l = 0
    print(len(test_data.edge_attr))
    for k in range(n_total):
        if not train_data.edge_attr[k] == 0:
            continue
        i = test_data.edge_index[0, k]
        j = test_data.edge_index[1, k]
        pos_i = train_data.x[i]
        pos_j = train_data.x[j]
        X_test[l] = torch.dist(pos_i, pos_j)
        y_test[l] = test_data.edge_attr[k]
        l += 1


    X_test = torch.unsqueeze(X_test, 1)
    y_pred = clf.predict(X_test)
    print(0, torch.count_nonzero(y_test==0), n_test)
    print(1, torch.count_nonzero(y_test==1), n_test)
    print(-1, torch.count_nonzero(y_test==-1), n_test)

    print(0, np.sum(y_pred==0), n_test)
    print(1, np.sum(y_pred==1), n_test)
    print(-1, np.sum(y_pred==-1), n_test)

    # evaluate the performance of the classifier
    auc_score = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return auc_score, acc, prec, rec, f1