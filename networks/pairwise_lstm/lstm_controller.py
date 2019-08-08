"""
The controller to train and test the pairwise_lstm network
"""

import numpy as np
from keras.models import Model
from keras.models import load_model

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils import TimeCalculator
from common.utils.ShortUtteranceConverter import create_data_lists
from common.utils.logger import *
from common.utils.paths import *
from .bilstm_2layer_dropout_plus_2dense import bilstm_2layer_dropout
from .core.data_gen import generate_test_data
from .core.pairwise_kl_divergence import pairwise_kl_divergence
from common.spectrogram.speaker_dev_selector import load_test_data

class LSTMController(NetworkController):

    def __init__(self, config, dev, best):
        super().__init__("pairwise_lstm", config, dev)
        self.network_file = self.name + "_100"
        self.best = best


    def get_network_name(self):
        return self.name + "_100"


    def train_network(self):

        n_classes = 100

        bilstm_2layer_dropout(
            self.network_file,
            self.config.get('train', 'pickle'),
            self.config.getint('pairwise_lstm', 'n_hidden1'),
            self.config.getint('pairwise_lstm', 'n_hidden2'),
            self.config.getint('pairwise_lstm', 'n_classes'),
            self.config.getint('pairwise_lstm', 'n_10_batches'),
            segment_size=self.config.getint('pairwise_lstm', 'seg_size')
        )

    def get_embeddings(self):
        short_utterance = self.config.getboolean('test', 'short_utterances')
        out_layer = self.config.getint('pairwise_lstm', 'out_layer')
        seg_size = self.config.getint('pairwise_lstm', 'seg_size')
        vec_size = self.config.getint('pairwise_lstm', 'vec_size')

        logger = get_logger('lstm', logging.INFO)
        logger.info('Run pairwise_lstm test')
        logger.info('out_layer -> ' + str(out_layer))
        logger.info('seg_size -> ' + str(seg_size))
        logger.info('vec_size -> ' + str(vec_size))

        # Load and prepare train/test data
        x_train, speakers_train, s_list_train = load_test_data(self.get_validation_train_data())
        x_test, speakers_test, s_list_test = load_test_data(self.get_validation_test_data())
        x_train, speakers_train, = prepare_data(x_train, speakers_train, seg_size)
        x_test, speakers_test = prepare_data(x_test, speakers_test, seg_size)

        x_list, y_list, s_list = create_data_lists(short_utterance, x_train, x_test,
                                                   speakers_train, speakers_test, s_list_train, s_list_test)

        # Prepare return values
        set_of_embeddings = []
        set_of_speakers = []
        speaker_numbers = []
        set_of_total_times = []
        
        if self.best:
            file_regex = self.get_network_name() + "*_best.h5"
        else:
            file_regex = self.get_network_name() + "*.h5"

        checkpoints = list_all_files(get_experiment_nets(), file_regex)

        # Values out of the loop
        metrics = ['accuracy', 'categorical_accuracy', ]
        loss = pairwise_kl_divergence
        custom_objects = {'pairwise_kl_divergence': pairwise_kl_divergence}
        optimizer = 'rmsprop'
        vector_size = vec_size

        # Fill return values
        for checkpoint in checkpoints:
            logger.info('Running checkpoint: ' + checkpoint)
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

            embeddings, speakers, num_embeddings = generate_embeddings(x_cluster_list, y_cluster_list, vector_size)

            # Fill the embeddings and speakers into the arrays
            set_of_embeddings.append(embeddings)
            set_of_speakers.append(speakers)
            speaker_numbers.append(num_embeddings)

            # Calculate the time per utterance
            time = TimeCalculator.calc_time_all_utterances(y_cluster_list, seg_size)
            set_of_total_times.append(time)

        logger.info('Pairwise_lstm test done.')
        return checkpoints, set_of_embeddings, set_of_speakers, speaker_numbers, set_of_total_times


def prepare_data(X,y, segment_size):
    x, speakers = generate_test_data(X, y, segment_size)

    # Reshape test data because it is an lstm
    return x.reshape(x.shape[0], x.shape[3], x.shape[2]), speakers
