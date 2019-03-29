from sklearn.metrics.cluster import adjusted_rand_score

def adjusted_rand_index(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)

if __name__ == '__main__':
    def flt_eq(x, y):
        return abs(x - y) < 1e-5

    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    assert flt_eq(adjusted_rand_index(y_true, y_pred), 1.0)

    y_true = [0, 0, 1, 2]
    y_pred = [0, 0, 1, 1]
    assert flt_eq(adjusted_rand_index(y_true, y_pred), 4/7)

    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 2]
    assert flt_eq(adjusted_rand_index(y_true, y_pred), 4/7)

    y_true = [0, 0, 0, 0]
    y_pred = [0, 1, 2, 3]
    assert flt_eq(adjusted_rand_index(y_true, y_pred), 0.0)

    y_true = [1, 2, 3, 3, 2, 1, 1, 3, 3, 1, 2, 2]
    y_pred = [3, 2, 3, 2, 2, 1, 1, 2, 3, 1, 3, 1]
    assert flt_eq(adjusted_rand_index(y_true, y_pred), 1/12)