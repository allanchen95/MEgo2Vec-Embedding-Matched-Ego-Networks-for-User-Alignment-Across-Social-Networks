from sklearn import metrics
import numpy as np

def cal_auc(y_true, prob_pred):
    return metrics.roc_auc_score(y_true, prob_pred)


def cal_f1(y_true, prob_pred):
    performance = metrics.precision_recall_fscore_support(y_true, prob_pred, average='binary')
    precision = performance[0]
    recall = performance[1]
    f1 = performance[2]
    return precision, recall, f1





