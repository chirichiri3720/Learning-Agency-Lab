import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


def quadratic_weighted_kappa(y_true, y_pred, min_rating=None, max_rating=None):
    """
    Calculates the Quadratic Weighted Kappa.
    """
    assert len(y_true) == len(y_pred)
    if min_rating is None:
        min_rating = min(min(y_true), min(y_pred))
    if max_rating is None:
        max_rating = max(max(y_true), max(y_pred))
    
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    O = confusion_matrix(y_true, y_pred, labels=np.arange(min_rating, max_rating + 1))
    
    num_ratings = len(O)
    num_scored_items = float(len(y_true))

    w = np.zeros((num_ratings, num_ratings))
    # for i in range(num_ratings):
    #     for j in range(num_ratings):
    #         w[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
		
    for i in range(num_ratings):
        for j in range(num_ratings):
            w[i, j] = ((y_true[i] - y_pred[j]) ** 2) / ((num_ratings - 1) ** 2)

    act_hist = np.zeros(num_ratings)
    for item in y_true:
        act_hist[item - min_rating] += 1

    pred_hist = np.zeros(num_ratings)
    for item in y_pred:
        pred_hist[item - min_rating] += 1

    E = np.outer(act_hist, pred_hist) / num_scored_items

    num = np.sum(w * O)
    den = np.sum(w * E)

    return 1.0 - (num / den)


# # QWKの計算
# qwk_score = quadratic_weighted_kappa(true_labels, pred_labels)
# print(f'Quadratic Weighted Kappa: {qwk_score}')