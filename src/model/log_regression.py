import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def log_regression(train_data, test_data):
    n_train = len(train_data.edge_index[0])
    X_train = torch.ones(n_train)
    y_train = torch.ones(n_train)

    for k in range(n_train):
        i = train_data.edge_index[0, k]
        j = train_data.edge_index[1, k]
        pos_i = train_data.x[i]
        pos_j = train_data.x[j]
        X_train[k] = torch.dist(pos_i, pos_j)
        y_train[k] = train_data.edge_attr[k]

    X_train = torch.unsqueeze(X_train, 1)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    n_test = len(test_data.edge_index[0])
    X_test = torch.ones(n_test)
    y_test = torch.ones(n_test)

    for k in range(n_test):
        i = test_data.edge_index[0, k]
        j = test_data.edge_index[1, k]
        pos_i = train_data.x[i]
        pos_j = train_data.x[j]
        X_test[k] = torch.dist(pos_i, pos_j)
        y_test[k] = test_data.edge_attr[k]

    X_test = torch.unsqueeze(X_test, 1)
    y_pred = clf.predict(X_test)

    # evaluate the performance of the classifier
    auc_score = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return auc_score, acc, prec, rec, f1