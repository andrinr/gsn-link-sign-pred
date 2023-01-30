import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def log_regression(train_data, test_data):

    X_train = torch.ones((len(train_data.edge_index[0]), 2 * train_data.x.shape[1]))
    y_train = torch.ones((len(train_data.edge_index[0]), 1))
    edge_list = train_data.edge_index.t().tolist()
    for k in range(len(edge_list)):
        i, j = edge_list[k]
        pos_i = train_data.x[i]
        pos_j = train_data.x[j]
        X_train[k] = torch.dist(pos_i, pos_j)
        y_train[k] = train_data.edge_attr[
            torch.where((train_data.edge_index[0] == i) & (train_data.edge_index[1] == j))]

    clf = LogisticRegression()
    y_train = torch.squeeze(y_train)
    clf.fit(X_train, y_train)

    X_test = torch.ones((len(test_data.edge_index[0]), 2 * train_data.x.shape[1]))
    y_test = torch.ones((len(test_data.edge_index[0]), 1))
    edge_list = train_data.edge_index.t().tolist()
    for k in range(len(edge_list)):
        i, j = edge_list[k]
        pos_i = train_data.x[i]
        pos_j = train_data.x[j]
        X_test[k] = torch.dist(pos_i, pos_j)
        y_test[k] = test_data.edge_attr[torch.where((test_data.edge_index[0] == i) & (test_data.edge_index[1] == j))]

    y_test = torch.squeeze(y_test)
    y_pred = clf.predict(X_test)

    # evaluate the performance of the classifier
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return acc, prec, rec, f1