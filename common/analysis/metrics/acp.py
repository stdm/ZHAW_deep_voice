from sklearn.metrics import confusion_matrix
import numpy as np


def average_cluster_purity(y_true, y_pred):
    """
    :param y_true: Ground truth speakers per utterance
    :param y_pred: Predicted speakers per utterance
    :return: The average cluster purity (ACP)
    """
    cm = confusion_matrix(y_pred, y_true)
    '''
    According to sklearn docs, the order of the parameters in the function above should be y_true and then y_pred.
    For this metric however, n_ij is equal to the number of observations predicted to be in group i but known to
    be in group j. So we have to change the order of the arguments.
    '''
    N = np.sum(cm)
    acp = 0.
    for cluster in cm:
        n_i = np.sum(cluster)
        if n_i != 0:
            p_i = 0.
            for n_ij in cluster:
                p_i += (n_ij * n_ij) / (n_i * n_i)
            acp += p_i * n_i
    acp /= N
    return acp
