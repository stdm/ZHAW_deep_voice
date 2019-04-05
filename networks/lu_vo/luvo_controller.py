"""
The controller to train and test the luvo network
"""

from common.network_controller import NetworkController
from common.utils.logger import *
from common.utils.paths import *
from common.spectogram.speaker_dev_selector import load_test_data, load_dev_test_data
from .network_training.spectrogram_cnn_590 import SpectrogramCnn590


class LuvoController(NetworkController):
    def __init__(self):
        super().__init__("luvo")
        self.checkpoint = self.name + ".pickle"
        self.logger = get_logger(self.name, logging.INFO)
        self.cnn = SpectrogramCnn590(get_experiment_nets(self.checkpoint))

    def train_network(self):
        self.cnn.create_and_train(get_speaker_pickle("speakers_590_clustering_without_raynolds_train"))

    def get_embeddings(self):

        if self.dev_mode:
            X_train, y_train = load_dev_test_data(self.get_validation_train_data(), self.val_data_size, 8)
            X_test, y_test = load_dev_test_data(self.get_validation_test_data(), self.val_data_size, 2)
        else:
            X_train, y_train = load_test_data(self.get_validation_train_data())
            X_test, y_test = load_test_data(self.get_validation_test_data())

        embeddings, speakers, num_embeddings = self.cnn.create_embeddings(X_train, y_train, X_test, y_test)

        # calc_MR(embeddings, speakers, num_embeddings, 'cosine')

        return [self.checkpoint], [embeddings], [speakers], [num_embeddings]

def calc_MR(X, y, num_speakers, linkage_metric):

    import numpy as np
    from scipy.spatial.distance import cdist
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import cdist

    X = cdist(X, X, linkage_metric)
    Z = linkage(X, method='complete', metric=linkage_metric)

    clusters = []
    for i in range(len(y)):
        clusters.append([i])

    for z in Z:
        clusters.append(clusters[int(z[0])] + clusters[int(z[1])])

    e = []
    e.append(np.ones(num_speakers*2, dtype=np.int))
    for z in Z:
        err = list(e[len(e)-1])
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
        MRs.append(misclassification_rate(num_speakers*2, sum(err)))

    print('MR=%f' % np.min(MRs))
    return MRs

def increase_error(indices, e, clusters):
    for i in indices:
        if i < len(e):
            e[i] = 1
        else:
            increase_error(clusters[i], e, clusters)

def misclassification_rate(N, e):
    MR = float(e)/N
    return MR