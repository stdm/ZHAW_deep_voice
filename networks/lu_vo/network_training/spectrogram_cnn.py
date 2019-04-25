"""
    This class creates, trains and safes a CNN and its parameters.

    It can also generate the embeddings of this network directly.

    The saved net-model can be used with clustering_network_foreign_conv in pairwise_kldiv, whereas this
    CNN serves as the "foreign conv".

    Work of Lukic and Vogt, adapted by Heusser
"""
import abc

import numpy as np
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet, TrainSplit

from common.clustering.generate_embeddings import generate_embeddings
from common.utils.logger import *
from common.utils.pickler import load, save
from networks.lu_vo.network_training.segment_batchiterator import SegmentBatchIterator
from common.utils.load_config import *
from common.utils.paths import *
from common.utils import TimeCalculator


class SpectrogramCnn:
    __metaclass__ = abc.ABCMeta

    def __init__(self, net_path):
        super().__init__()
        self.logger = get_logger("luvo", logging.INFO)
        self.net_path = net_path
        self.config = load_config(None, join(get_common(), 'config.cfg'))

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
        net = self._create_net(self.create_paper(x.shape[1]))

        # Set new batch iterator
        net.batch_iterator_train = SegmentBatchIterator(batch_size=128, config=self.config)
        net.batch_iterator_test = SegmentBatchIterator(batch_size=128, config=self.config)
        net.train_split = TrainSplit(eval_size=0)

        # Train the network
        self.logger.info("Fitting...")
        net.fit(x, y)

        # Comments from old spectrogram_cnn_100 implementation, don't delete yet if eventually needed later
        # net.load_params_from('../data/experiments/paper/networks/net_100_81_not_reynolds.pickle');
        # net.save_params_to('../../data/experiments/paper/networks/net_100_81_not_reynolds.pickle');
        # network_helper.save(net, '../../data/experiments/paper/networks/net_100_81_not_reynolds.pickle')
        save(net, self.net_path)

    def create_embeddings(self, X_train, y_train, X_test, y_test):
        x_train_cluster, y_train_cluster = self._generate_cluster_data(X_train, y_train)
        x_test_cluster, y_test_cluster = self._generate_cluster_data(X_test, y_test)

        # Load the network and add Batchiterator
        net = load(self.net_path)
        net.batch_iterator_test = BatchIterator(batch_size=128)

        # Predict the output
        # predict = prepare_predict(net)
        # output_train = predict(x_train_cluster)
        # output_test = predict(x_test_cluster)

        output_train = net.predict_proba(x_train_cluster)
        output_test = net.predict_proba(x_test_cluster)

        embeddings, speakers, number_embeddings =\
            generate_embeddings(output_train, output_test, y_train_cluster, y_test_cluster, output_train.shape[1])

        #Calculate the time per utterance
        time = TimeCalculator.calc_time_long_short_utterances(y_train_cluster, y_test_cluster,
                                                              self.config.getint('luvo', 'seg_size'))

        return embeddings, speakers, number_embeddings, time

    def _create_net(self, paper):
        # Setup the neural network
        net = NeuralNet(
            layers=paper,

            # learning rate parameters
            update_learning_rate=self.config.getfloat('luvo', 'update_learning_rate'),
            update_momentum=self.config.getfloat('luvo', 'update_momentum'),
            regression=self.config.getboolean('luvo', 'regression'),

            max_epochs=self.config.getint('luvo', 'num_epochs'),
            verbose=self.config.getint('luvo', 'verbose'),
        )

        return net

    def _generate_cluster_data(self, X, y):
        seg_size = self.config.getint('luvo', 'seg_size')
        spectrogram_height = self.config.getint('luvo', 'spectrogram_height')
        X_cluster = np.zeros((10000, 1, spectrogram_height, seg_size), dtype=np.float32)
        y_cluster = []

        pos = 0
        for i in range(len(X)):
            spectrogram = self._crop_spectrogram(X[i, 0])

            for j in range(int(spectrogram.shape[1] / seg_size)):
                y_cluster.append(y[i])
                seg_idx = j * seg_size
                X_cluster[pos, 0] = spectrogram[:, seg_idx:seg_idx + seg_size]
                pos += 1

        return X_cluster[0:len(y_cluster)], np.asarray(y_cluster, dtype=np.int32)

    def _crop_spectrogram(self, spectrogram):
        zeros = 0
        for x in spectrogram[0]:
            if x == 0.0:
                zeros += 1
            else:
                zeros = 0
        return spectrogram[0:self.config.getint('luvo', 'spectrogram_height'), 0:spectrogram.shape[1] - zeros]

'''
def prepare_predict(net):
    input_var = T.tensor4('x')
    conv_network = net.get_all_layers()[7]
    y = get_output(conv_network, input_var)
    return theano.function([input_var], y)
'''
