"""
    A suite for calculating the MR.
    The main two methods "exported" to the outside is:
    - calc_MR(X, y, num_speakers, linkage_metric)
    - generate_X(train_output, test_output, train_speakers, test_speakers, neuron_number)

    Work based of Lukic and Vogt.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage


def calc_MR(X, y, num_speakers, linkage_metric):
    # cityblock, braycurtis,
    from scipy.spatial.distance import cdist
    X = cdist(X, X, linkage_metric)
    Z = linkage(X, method='complete', metric=linkage_metric)

    clusters = []
    for i in range(len(y)):
        clusters.append([i])

    for z in Z:
        clusters.append(clusters[int(z[0])] + clusters[int(z[1])])

    e = []
    e.append(np.ones(len(y), dtype=np.int))
    for z in Z:
        err = list(e[len(e) - 1])
        idx1 = int(z[0])
        idx2 = int(z[1])
        if idx1 >= len(y) or idx2 >= len(y) or y[idx1] != y[idx2]:
            indices = clusters[idx1] + clusters[idx2]
            increase_error(indices, err, clusters)
        else:
            err[idx1] = 0
            err[idx2] = 0
        e.append(err)

    MRs = []
    for err in e:
        MRs.append(misclassification_rate(len(y), sum(err)))

    print('MR=%f' % np.min(MRs))
    return MRs


def generate_embedings(train_output, test_output, train_speakers, test_speakers, neuron_number):
    num_speakers = len(set(test_speakers))

    X_train, y_train = extract_vectors(num_speakers, neuron_number, train_output, train_speakers)
    X_test, y_test = extract_vectors(num_speakers, neuron_number, test_output, test_speakers)
    X = []
    X.extend(X_train)
    X.extend(X_test)
    y = []
    y.extend(y_train)
    y.extend(y_test)

    return X, y, num_speakers


def misclassification_rate(N, e):
    MR = float(e) / N
    return MR


def extract_vectors(num_speakers, vec_size, vectors, y):
    X = np.zeros((num_speakers, vec_size))
    for i in range(num_speakers):
        indices = np.where(y == i)[0]
        outputs = np.take(vectors, indices, axis=0)
        for o in outputs:
            X[i] = np.add(X[i], o)
        X[i] = np.divide(X[i], len(outputs))
    return X, set(y)


def increase_error(indices, e, clusters):
    for i in indices:
        if i < len(e):
            e[i] = 1
        else:
            increase_error(clusters[i], e, clusters)
