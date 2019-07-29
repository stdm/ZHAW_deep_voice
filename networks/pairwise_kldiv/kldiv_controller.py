"""
The controller to train and test the pairwise_kldiv network
"""
from keras import Model
from keras.engine.saving import load_model

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils import TimeCalculator
from common.utils.ShortUtteranceConverter import create_data_lists
from common.utils.logger import *
from networks.pairwise_kldiv.keras_network_training.clustering_network import create_and_train, get_experiment_nets, \
    get_speaker_pickle
#from .network_training.clustering_network import create_and_train
from networks.pairwise_kldiv.keras_network_training.network_factory import create_network_n_speakers
from networks.pairwise_lstm.core.data_gen import generate_test_data
from networks.pairwise_lstm.core.pairwise_kl_divergence import pairwise_kl_divergence, orig_pairwise_kl_divergence
from .analysis.run_analysis_network import run_analysis_network

#from .network_training.network_factory import *
from common.spectrogram.speaker_dev_selector import load_test_data

import numpy as np


class KLDivController(NetworkController):
    def __init__(self, config):
        super().__init__("pairwise_kldiv", config)
        #self.checkpoints = ["pairwise_kldiv_100.pickle"]
        self.checkpoints = ["pairwise_kldiv_100.h5"]

    def train_network(self):
        net_file = get_experiment_nets(self.checkpoints[0])
        train_file = get_speaker_pickle(self.config.get('train', 'pickle'),)

        '''
        #import original create_and_train function and network_factory if this should be used again
        create_and_train(num_epochs=config.getint('pairwise_kldiv', 'num_epochs'),
                         batch_size=config.getint('pairwise_kldiv', 'batch_size'),
                         network_params_file_in=None,
                         network_params_file_out=net_file,
                         train_file=train_file,
                         epoch_batches=config.getint('pairwise_kldiv', 'epoch_batches'),
                         network_fun=create_network_470_speakers, with_validation=False)
        '''

        num_epochs = self.config.getint('pairwise_kldiv', 'num_epochs')
        batch_size = self.config.getint('pairwise_kldiv', 'batch_size')
        epoch_batches = self.config.getint('pairwise_kldiv', 'epoch_batches')
        network = create_network_n_speakers(100, self.config)

        create_and_train(num_epochs=num_epochs,
                         batch_size=batch_size,
                         network_params_file_out=net_file,
                         train_file=train_file,
                         epoch_batches=epoch_batches,
                         network=network)

    '''
    def get_embeddings(self):
        short_utterance = self.config.getboolean('validation', 'short_utterances')
        logger = get_logger('kldiv', logging.INFO)
        logger.info('Run pairwise_kldiv')
        checkpoints = self.checkpoints

        X_train, y_train, s_list_train = load_test_data(self.get_validation_train_data())
        X_test, y_test, s_list_test = load_test_data(self.get_validation_test_data())

        x_list, y_list, s_list = create_data_lists(short_utterance, X_train, X_test,
                                                   y_train, y_test, s_list_train, s_list_test)

        # Prepare return value
        set_of_embeddings = []
        set_of_speakers = []
        set_of_num_embeddings = []
        set_of_total_times = []

        for checkpoint in checkpoints:
            logger.info('Run checkpoint: ' + checkpoint)
            network_file = get_experiment_nets(checkpoint)

            x_cluster_list = []
            y_cluster_list = []
            for x, y, s in zip(x_list, y_list, s_list):
                x_cluster, y_cluster = run_analysis_network(network_file, x, y, s)
                x_cluster_list.append(x_cluster)
                y_cluster_list.append(y_cluster)

            embeddings, speakers, num_embeddings =\
                generate_embeddings(x_cluster_list, y_cluster_list, x_cluster_list[0].shape[1])
            # Fill return values
            set_of_embeddings.append(embeddings)
            set_of_speakers.append(speakers)
            set_of_num_embeddings.append(num_embeddings)

            # Calculate the time per utterance
            time = TimeCalculator.calc_time_all_utterances(y_cluster_list, config.getint('pairwise_kldiv', 'seg_size'))
            set_of_total_times.append(time)

        return checkpoints, set_of_embeddings, set_of_speakers, set_of_num_embeddings, set_of_total_times
    '''

    def get_embeddings(self):
        checkpoints = ["pairwise_kldiv_100.h5"]
        short_utterance = self.config.getboolean('validation', 'short_utterances')
        out_layer = self.config.getint('pairwise_kldiv', 'out_layer')
        seg_size = self.config.getint('pairwise_kldiv', 'seg_size')
        logger = get_logger('kldiv', logging.INFO)
        logger.info('Run pairwise_kldiv')

        # Load and prepare train/test data
        X_train, y_train, s_list_train = load_test_data(self.get_validation_train_data())
        X_test, y_test, s_list_test = load_test_data(self.get_validation_test_data())
        X_train, y_train, = self._prepare_data(X_train, y_train, seg_size)
        X_test, y_test = self._prepare_data(X_test, y_test, seg_size)

        x_list, y_list, s_list = create_data_lists(short_utterance, X_train, X_test,
                                                   y_train, y_test, s_list_train, s_list_test)

        # Prepare return value
        set_of_embeddings = []
        set_of_speakers = []
        set_of_num_embeddings = []
        set_of_total_times = []

        # Values out of the loop
        metrics = ['accuracy']
        loss = orig_pairwise_kl_divergence
        custom_objects = {'orig_pairwise_kl_divergence': orig_pairwise_kl_divergence}
        optimizer = 'adadelta'

        for checkpoint in checkpoints:
            logger.info('Run checkpoint: ' + checkpoint)
            # Load and compile the trained network
            network_file = get_experiment_nets(checkpoint)
            model_full = load_model(network_file, custom_objects=custom_objects)
            model_full.compile(loss=loss, optimizer=optimizer, metrics=metrics)

            # Get a Model with the embedding layer as output and predict
            model_partial = Model(inputs=model_full.input, outputs=model_full.layers[out_layer].output)

            x_cluster_list = []
            y_cluster_list = []
            for x, y, s in zip(x_list, y_list, s_list):
                x_cluster = np.asarray(model_partial.predict(x))
                x_cluster_list.append(x_cluster)
                y_cluster_list.append(y)

            embeddings, speakers, num_embeddings =\
                generate_embeddings(x_cluster_list, y_cluster_list, x_cluster_list[0].shape[1])

            # Fill return values
            set_of_embeddings.append(embeddings)
            set_of_speakers.append(speakers)
            set_of_num_embeddings.append(num_embeddings)

            # Calculate the time per utterance
            time = TimeCalculator.calc_time_all_utterances(y_cluster_list,
                                                           self.config.getint('pairwise_kldiv', 'seg_size'))
            set_of_total_times.append(time)

        return checkpoints, set_of_embeddings, set_of_speakers, set_of_num_embeddings, set_of_total_times


    def _prepare_data(self, X, y, segment_size):
        x, speakers = generate_test_data(X, y, segment_size)

        # Reshape test data
        return x.reshape(x.shape[0], x.shape[1], x.shape[3], x.shape[2]), speakers
