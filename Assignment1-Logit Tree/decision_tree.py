#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
# In[2]:


def entropy(y):
    vals, counts = np.unique(y, return_counts=True)
    ps = counts / len(y)
    ent = 0
    for p in ps:
        if(p > 0):
            ent += p * np.log2(p)
    ent = -1 * ent
    return ent


def entropy(y):
    vals, counts = np.unique(y, return_counts=True)
    ps = counts / len(y)
    ent = 0
    for p in ps:
        if(p > 0):
            ent += p * np.log2(p)
    ent = -1 * ent
    return ent


class Tree_Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.value = value
        self.feature = feature
        self.left = left
        self.right = right
        self.threshold = threshold

    def is_leaf_node(self):
        if(self.value is None):
            return False
        return True


class Tree_Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.value = value
        self.feature = feature
        self.left = left
        self.right = right
        self.threshold = threshold

    def is_leaf_node(self):
        if(self.value is None):
            return False
        return True


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_feats=None, logistic_regression_model=LogisticRegression(max_iter=100), multi_feature=False):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        self.logistic_regression_model = logistic_regression_model
        self.depth_wise_nodes_dict = {}
        self.depth_wise_threshold_dict = {}
        self.multi_feature = multi_feature

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(
            self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            if(n_samples == 0):
                return Tree_Node(value=0)

            leaf_value = self._most_common_label(y)
            return Tree_Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain after logistic regression

        if (not self.multi_feature):
            best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        else:
            best_feat, best_thresh = self._best_criteria_two_features(
                X, y, feat_idxs)

        # for visualizing decision tree
        if(depth not in self.depth_wise_nodes_dict):
            self.depth_wise_nodes_dict[depth] = [best_feat]

        else:
            self.depth_wise_nodes_dict[depth].append(best_feat)

        if(depth not in self.depth_wise_threshold_dict):
            self.depth_wise_threshold_dict[depth] = [best_thresh]

        else:
            self.depth_wise_threshold_dict[depth].append(best_thresh)

        if (self.multi_feature):
            best_feat = best_feat[1]
            best_thresh = best_thresh[1]

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Tree_Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_accuracy = -1
        best_gain = -1
        split_thresh = None
        best_feature_idx = None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            X_col_train, X_col_test, y_col_train, y_col_test = train_test_split(
                X_column, y, test_size=0.2, random_state=1234
            )
            X_col_train = X_col_train.reshape(-1, 1)
            X_col_test = X_col_test.reshape(-1, 1)

            if(len(np.unique(y_col_train)) == 1):
                best_feature_idx = feat_idx
                break

            self.logistic_regression_model.fit(X_col_train, y_col_train)
            curr_accuracy = self.logistic_regression_model.score(
                X_col_test, y_col_test)

            if curr_accuracy > best_accuracy:
                best_accuracy = curr_accuracy
                best_feature_idx = feat_idx

        X_column = X[:, best_feature_idx]
        thresholds = np.unique(X_column)

        for threshold in thresholds:
            gain = self._information_gain(y, X_column, threshold)

            if gain > best_gain:
                best_gain = gain
                split_thresh = threshold

        return best_feature_idx, split_thresh

    def _best_criteria_two_features(self, X, y, feat_idxs):
        best_gain = -1
        split_thresh1 = None
        split_thresh2 = None

        accuracy_store = {}
        max_accuracy = -1
        max_accuracy_pair = []

        for feat_idx1 in feat_idxs:
            for feat_idx2 in feat_idxs:

                if((feat_idx1, feat_idx2) in accuracy_store or (feat_idx2, feat_idx1) in accuracy_store or feat_idx1 == feat_idx2):
                    continue

                X_column = X[:, [feat_idx1, feat_idx2]]
                X_col_train, X_col_test, y_col_train, y_col_test = train_test_split(
                    X_column, y, test_size=0.2, random_state=1234
                )
                X_col_train = X_col_train.reshape(-1, 1)
                X_col_test = X_col_test.reshape(-1, 1)

                if(len(np.unique(y_col_train)) == 1):
                    max_accuracy_pair = [feat_idx1, feat_idx2]
                    accuracy_store[(feat_idx1, feat_idx2)] = 1
                    max_accuracy = 1
                    break

                X_col_train = X_col_train.reshape(y_col_train.shape[0], -1)
                X_col_test = X_col_test.reshape(y_col_test.shape[0], -1)

                self.logistic_regression_model.fit(X_col_train, y_col_train)
                curr_accuracy = self.logistic_regression_model.score(
                    X_col_test, y_col_test)

                accuracy_store[(feat_idx1, feat_idx2)] = curr_accuracy

                if curr_accuracy > max_accuracy:
                    max_accuracy = curr_accuracy
                    max_accuracy_pair = [feat_idx1, feat_idx2]

        X_column1 = X[:, max_accuracy_pair[0]]
        X_column2 = X[:, max_accuracy_pair[1]]

        thresholds1 = np.unique(X_column1)
        thresholds2 = np.unique(X_column2)

        for thresholda in thresholds1:
            for thresholdb in thresholds2:

                gain1 = self._information_gain(y, X_column1, thresholda)
                gain2 = self._information_gain(y, X_column2, thresholdb)

                if gain1*gain2 > best_gain:
                    best_gain = gain2*gain1
                    split_thresh1 = thresholda
                    split_thresh2 = thresholdb

        return max_accuracy_pair, [split_thresh1, split_thresh2]

    def visualize_tree(self):
        """ this function should be called after fitting
        """

        depth = 0
        print("the tree is splitted according to the following features")

        while (depth in self.depth_wise_nodes_dict):
            print("Level ", depth)
            print("Nodes are : ")
            for idx in range(len(self.depth_wise_nodes_dict[depth])):
                print(self.depth_wise_nodes_dict[depth][idx], "( Threshold : " + str(
                    self.depth_wise_threshold_dict[depth][idx]) + " ) ", end=" ")
            print()
            depth += 1

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
