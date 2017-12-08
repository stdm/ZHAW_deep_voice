"""
    This class creates, trains and safes a CNN and its parameters.

    It can also generate the embeddings of this network directly.

    The saved net-model can be used with clustering_network_foreign_conv in pairwise_kldiv, whereas this
    CNN serves as the "foreign conv".

    Work of Lukic and Vogt, adapted by Heusser
"""
import abc

import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import get_output
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet, TrainSplit

from common.clustering.generate_embeddings import generate_embeddings
from common.utils.logger import *
from common.utils.pickler import load, save
from .segment_batchiterator import SegmentBatchIterator
from .. import settings


class SpectrogramCnn:
    __metaclass__ = abc.ABCMeta

    def __init__(self, net_path):
        super().__init__()
        self.logger = get_logger("luvo", logging.INFO)
        self.net_path = net_path

    @abc.abstractmethod
    def create_paper(self, shape):
        """Creates the paper (layer definition) for this Spectrogram CNN"""
        return

    # training_data = '../data/training/speaker_pickles/train_data_630.pickle'
    # save_pickle = '../data/experiments/speaker_identification_for_clustering/networks/net_630.pickle'
    def create_and_train(self, training_data):
        # Load training data
        x, y, speaker_names = load(training_data)

        # Create network
        net = create_net(self.create_paper(x.shape[1]))

        # Set new batch iterator
        net.batch_iterator_train = SegmentBatchIterator(batch_size=128)
        net.batch_iterator_test = SegmentBatchIterator(batch_size=128)
        net.train_split = TrainSplit(eval_size=0)

        # Train the network
        self.logger.info("Fitting...")
        net.fit(x, y)

        # Comments from old spectrogram_cnn_100 implementation, don't delete yet if eventually needed later
        # net.load_params_from('../data/experiments/paper/networks/net_100_81_not_reynolds.pickle');
        # net.save_params_to('../../data/experiments/paper/networks/net_100_81_not_reynolds.pickle');
        # network_helper.save(net, '../../data/experiments/paper/networks/net_100_81_not_reynolds.pickle')
        save(net, self.net_path)

    def create_embeddings(self, train_data, test_data):
        x_train_cluster, y_train_cluster = load_and_prepare_data(train_data)
        x_test_cluster, y_test_cluster = load_and_prepare_data(test_data)

        # Load the network and add Batchiterator
        net = load(self.net_path)
        net.batch_iterator_test = BatchIterator(batch_size=128)

        # Predict the output
        # predict = prepare_predict(net)
        # output_train = predict(x_train_cluster)
        # output_test = predict(x_test_cluster)

        output_train = net.predict_proba(x_train_cluster)
        output_test = net.predict_proba(x_test_cluster)

        return generate_embeddings(output_train, output_test, y_train_cluster, y_test_cluster, output_train.shape[1])

def prepare_predict(net):
    input_var = T.tensor4('x')
    conv_network = net.get_all_layers()[7]
    y = get_output(conv_network, input_var)
    return theano.function([input_var], y)

def create_net(paper):
    # Setup the neural network
    net = NeuralNet(
        layers=paper,

        # learning rate parameters
        update_learning_rate=0.001,
        update_momentum=0.9,
        regression=False,

        max_epochs=1000,
        verbose=1,
    )

    return net


def load_and_prepare_data(data_path):
    # Load and generate test data
    x, y, s_list = load(data_path)
    return generate_cluster_data(x, y)


def generate_cluster_data(X, y):
    X_cluster = np.zeros((10000, 1, settings.FREQ_ELEMENTS, settings.ONE_SEC), dtype=np.float32)
    y_cluster = []

    pos = 0
    for i in range(len(X)):
        spectrogram = crop_spectrogram(X[i, 0])

        for j in range(int(spectrogram.shape[1] / settings.ONE_SEC)):
            y_cluster.append(y[i])
            seg_idx = j * settings.ONE_SEC
            X_cluster[pos, 0] = spectrogram[:, seg_idx:seg_idx + settings.ONE_SEC]
            pos += 1

    return X_cluster[0:len(y_cluster)], np.asarray(y_cluster, dtype=np.int32)


def crop_spectrogram(spectrogram):
    zeros = 0
    for x in spectrogram[0]:
        if x == 0.0:
            zeros += 1
        else:
            zeros = 0
    return spectrogram[0:settings.FREQ_ELEMENTS, 0:spectrogram.shape[1] - zeros]
