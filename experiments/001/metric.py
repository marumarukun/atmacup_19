import numpy as np
from sklearn.metrics import roc_auc_score


def score(y_true, y_pred):
    """
    ROC AUCスコアを計算する関数

    Parameters:
    -----------
    y_true : array-like
        実際のラベル（0または1）
    y_pred : array-like
        予測確率

    Returns:
    --------
    float
        ROC AUCスコア
    """
    return roc_auc_score(y_true, y_pred)


def macro_auc_score(y_true, y_pred):
    """
    マクロ平均ROC AUCスコアを計算する関数
    4つのクラスに対するROC AUCの平均を返します

    Parameters:
    -----------
    y_true : array-like of shape (n_samples, n_classes)
        実際のラベル（one-hotエンコーディング形式）
    y_pred : array-like of shape (n_samples, n_classes)
        各クラスの予測確率

    Returns:
    --------
    float
        マクロ平均ROC AUCスコア
    """
    n_classes = y_true.shape[1]
    auc_scores = []

    for i in range(n_classes):
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        auc_scores.append(auc)

    return np.mean(auc_scores)
