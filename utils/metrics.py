# utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def multilabel_accuracy(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    per_class_accuracy = (y_pred_bin == y_true).sum(axis=0) / y_true.shape[0]
    return per_class_accuracy

def auc_roc_per_class(y_true, y_pred):
    results = []
    for i in range(y_true.shape[1]):
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
        except:
            score = float("nan")
        results.append(score)
    return results
