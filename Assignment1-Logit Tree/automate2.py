import decision_tree
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

datasets = ['fetal_health', 'cervical_cancer', 'banking_dataset' ]

for dataset in datasets:
    file_path = './Datasets/' + dataset + '/preprocessed_data.csv'

    data = pd.read_csv(file_path)
    data = data.sample(frac=1).reset_index()
    data.drop('index', axis=1, inplace=True)
    features = list(data.columns.values[:-1])
    data = data.to_numpy(dtype=None, copy=False)
    X, y = np.split(data, [len(features)], axis=1)
    y = y.T[0]

    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    acc = []
    prec = []
    recall = []
    f1 = []
    for train_index, test_index in kf.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        clf = decision_tree.DecisionTree(max_depth=20)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc.append(accuracy_score(y_test, y_pred))
        prec.append(precision_score(y_test, y_pred, average = 'weighted'))
        recall.append(recall_score(y_test, y_pred, average = 'weighted'))
        f1.append(f1_score(y_test, y_pred, average = 'weighted'))

    print("Accuracy_K:", acc)
    print("Precision_K:", prec)
    print("Recall_K:", recall)
    print("F1_Score_K:", f1)

    print()
    # print("ROC-AUC-Score for dataset " + dataset + ' is: ' + str(roc))


# kf = KFold(n_splits=5)
# kf.get_n_splits(X)

# acc = []
# prec = []
# recall = []
# f1 = []
# for train_index, test_index in kf.split(X):
#     X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
#     clf = DecisionTree(max_depth=20)
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)
#     acc.append(accuracy_score(y_test, y_pred))
#     prec.append(precision_score(y_test, y_pred, average = 'weighted'))
#     recall.append(recall_score(y_test, y_pred, average = 'weighted'))
#     f1.append(f1_score(y_test, y_pred, average = 'weighted'))

# print("Accuracy_K:", acc)
# print("Precision_K:", prec)
# print("Recall_K:", recall)
# print("F1_Score_K:", f1)
