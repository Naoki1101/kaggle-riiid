from sklearn import metrics


def auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)
