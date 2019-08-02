from sklearn.metrics.cluster import adjusted_rand_score


def adjusted_rand_index(y_true, y_pred):
    """
    :param y_true: Ground truth speakers per utterance
    :param y_pred: Predicted speakers per utterance
    :return: The adjusted rand index (ARI)
    """
    return adjusted_rand_score(y_true, y_pred)
