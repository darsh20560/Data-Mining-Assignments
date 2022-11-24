import pandas as pd
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import decision_tree
import matplotlib.pyplot as plt
import scipy

datasets = ['banking_dataset', 'fetal_health', 'cervical_cancer']

for dataset in datasets:
    file_path = './Datasets/' + dataset + '/preprocessed_data.csv'

    data = pd.read_csv(file_path)
    data = data.sample(frac=1).reset_index()
    data.drop('index', axis=1, inplace=True)
    features = list(data.columns.values[:-1])
    data = data.to_numpy(dtype=None, copy=False)
    X, y = np.split(data, [len(features)], axis=1)
    y = y.T[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = decision_tree.DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)
    clf.visualize_tree()

    y_pred_s = clf.predict(X_test)
    report = classification_report(y_test, y_pred_s)
    print(report)
    acc = accuracy_score(y_test, y_pred_s)
    rec = recall_score(y_test, y_pred_s, average='weighted')
    pre = precision_score(y_test, y_pred_s, average='weighted')
    f1 = f1_score(y_test, y_pred_s, average='weighted')

    print("Single Attribute Accuracy for dataset " + dataset + ' is: ' + str(acc))
    print("Single Attribute Recall for dataset " + dataset + ' is: ' + str(rec))
    print("Single Attribute Precision for dataset " +
          dataset + ' is: ' + str(pre))
    print("Single Attribute F1-Score for dataset " + dataset + ' is: ' + str(f1))


    print()

    clf2 = decision_tree.DecisionTree(max_depth=10, multi_feature=True)
    clf2.fit(X_train, y_train)
    clf2.visualize_tree()

    y_pred_m = clf2.predict(X_test)
    report = classification_report(y_test, y_pred_m)
    print(report)
    acc = accuracy_score(y_test, y_pred_m)
    rec = recall_score(y_test, y_pred_m, average='weighted')
    pre = precision_score(y_test, y_pred_m, average='weighted')
    f1 = f1_score(y_test, y_pred_m, average='weighted')

    print("Multi Attribute Accuracy for dataset " + dataset + ' is: ' + str(acc))
    print("Multi Attribute Recall for dataset " + dataset + ' is: ' + str(rec))
    print("Multi Attribute Precision for dataset " + dataset + ' is: ' + str(pre))
    print("Multi Attribute F1-Score for dataset " + dataset + ' is: ' + str(f1))
    print()

    t_ttest_ind, p_ttest_ind = scipy.stats.ttest_ind(y_pred_m, y_pred_s)
    t_f_oneway, p_f_oneway = scipy.stats.f_oneway(y_pred_m, y_pred_s)
    t_kruskal, p_kruskal = scipy.stats.kruskal(y_pred_m, y_pred_s)

    print(t_ttest_ind, " ", p_ttest_ind)
    print(t_f_oneway, " ", p_f_oneway)
    print(t_kruskal, " ", p_kruskal)
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
