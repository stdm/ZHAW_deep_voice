'''
This file calculates the MR for a given pair of Cluster Outputs, this code is altered slightly from
the Bachelor thesis  of Vogt and Lukic (2016).

'''
import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage

from common.utils.paths import *
from .new_mr_calculation import generate_X


def misclassification_rate(N, e):
    MR = float(e) / N
    # print float(e)
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


def calc_MR(X, y, num_speakers, linkage_metric):
    '''
    Generates the Hierarchical Cluster based on the Output of generate_speaker_utterances, with the supplied linkage Metric.
    Based on the Clustering the MR will be Calculated. The MR calculation perceives an embedding
    as wrong as soon as there is a non matching embedding in the cluster.
    '''

    # cityblock, braycurtis,
    from scipy.spatial.distance import cdist
    X = cdist(X, X, linkage_metric)
    Z = linkage(X, method='complete', metric=linkage_metric)
    clusters = []
    for i in range(len(y)):
        clusters.append([i])
    i = 0
    for z in Z:
        clusters.append(clusters[int(z[0])] + clusters[int(z[1])])

        i += 1

    e = []
    e.append(np.ones(len(y), dtype=np.int))
    print(Z.shape)
    for z in Z:
        err = list(e[len(e) - 1])
        idx1 = int(z[0])
        idx2 = int(z[1])
        # if idx1 < len(y) and idx2 < len(y) and y[idx1] != y[idx2]:
        #     print y[idx1]
        #     print y[idx2]
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


def generate_speaker_utterances(train_output, test_output, train_speakers, test_speakers, neuron_number):
    '''
    creates for all Speakers the the Average over their embedings to create an uterance 
    for a each speaker in the Train set (8 sentences), test set (2 sentences)
    the two lists get concatinated to create num_speakers *2 list.
    '''
    num_speakers = len(set(test_speakers))
    print(num_speakers)
    X_train, y_train = extract_vectors(num_speakers, neuron_number, train_output, train_speakers)
    X_test, y_test = extract_vectors(num_speakers, neuron_number, test_output, test_speakers)
    X = []
    X.extend(X_train)
    X.extend(X_test)
    y = []
    y.extend(y_train)
    y.extend(y_test)

    return X, y, num_speakers


def load_data(train_file, test_file):
    '''
    
    '''
    with open(get_speaker_pickle(train_file), 'rb') as f:
        train_output, train_speakers, train_speaker_names = pickle.load(f)
    with open(get_speaker_pickle(test_file), 'rb') as f:
        test_output, test_speakers, test_speaker_names = pickle.load(f)
    return train_output, test_output, train_speakers, test_speakers


def evaluate_mr(train_file, test_file, neuron_number, label):
    train_output, test_output, train_speakers, test_speakers = load_data(train_file, test_file)
    X, y, num_speakers = generate_X(train_output, test_output, train_speakers, test_speakers, neuron_number)
    MRs = calc_MR(X, y, num_speakers, 'cosine')
    plt.plot(MRs, label=label, linewidth=2)


if __name__ == "__main__":
    evaluate_mr('train_cluster_out_40sp__256_500_100sp',
                'test_cluster_out_40sp__256_500_100sp', 256, '40sp')

    evaluate_mr('cluster_output_train_60sp_2017-05-25_21-23-42',
                'cluster_output_test_60sp_2017-05-25_21-23-42', 256, '60sp')

    evaluate_mr('cluster_output_train_80sp_2017-05-25_21-23-42',
                'cluster_output_test_80sp_2017-05-25_21-23-42', 256, '60sp')

    plt.xlabel('Clusters')
    plt.ylabel('Misclassification Rate (MR)')
    plt.grid()
    plt.legend(loc='lower right', shadow=False)
    plt.ylim(0, 1)
    plt.show()
