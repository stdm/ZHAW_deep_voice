from sklearn.metrics import confusion_matrix
import numpy as np

def average_cluster_purity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    N = np.sum(cm)
    acp = 0.
    for cluster in cm:
        n_i = np.sum(cluster)
        p_i = 0.
        for n_ij in cluster:
            p_i += (n_ij * n_ij) / (n_i * n_i)
        acp += p_i * n_i
    acp /= N
    return acp



if __name__ == '__main__':
    def flt_eq(x, y):
        return abs(x - y) < 1e-5

    y_true = [0, 1]
    y_pred = [1, 0]
    assert flt_eq(acp(y_true, y_pred), 1.0)

    y_true = [0, 0 ,1]
    y_pred = [1, 0, 0]
    assert flt_eq(acp(y_true, y_pred), 2/3)

    y_true = [0, 1, 2, 1]
    y_pred = [0, 1, 1, 2]
    assert flt_eq(acp(y_true, y_pred), 0.75)