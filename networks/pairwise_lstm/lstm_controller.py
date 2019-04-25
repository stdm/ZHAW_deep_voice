"""
The controller to train and test the pairwise_lstm network
"""

import numpy as np
from keras.models import Model
from keras.models import load_model

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils import TimeCalculator
from common.utils.logger import *
from common.utils.paths import *
from .bilstm_2layer_dropout_plus_2dense import bilstm_2layer_dropout
from .core.data_gen import generate_test_data
from .core.pairwise_kl_divergence import pairwise_kl_divergence
from common.utils.load_config import *
from common.spectrogram.speaker_dev_selector import load_test_data, load_dev_test_data

class LSTMController(NetworkController):
    def __init__(self):
        super().__init__("pairwise_lstm")
        self.network_file = self.name + "_100"
        self.config = load_config(None, join(get_common(), 'config.cfg'))

    def train_network(self):
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

        # Prepare return values
        set_of_embeddings = []
        set_of_speakers = []
        speaker_numbers = []
        set_of_total_times = []
        checkpoints = list_all_files(get_experiment_nets(), "^pairwise_lstm.*\.h5")

        # Values out of the loop
        metrics = ['accuracy', 'categorical_accuracy', ]
        loss = pairwise_kl_divergence
        custom_objects = {'pairwise_kl_divergence': pairwise_kl_divergence}
        optimizer = 'rmsprop'
        vector_size = vec_size #256 * 2

        # Fill return values
        for checkpoint in checkpoints:
            logger.info('Running checkpoint: ' + checkpoint)
            # Load and compile the trained network
            network_file = get_experiment_nets(checkpoint)
            model_full = load_model(network_file, custom_objects=custom_objects)
            model_full.compile(loss=loss, optimizer=optimizer, metrics=metrics)

            # Get a Model with the embedding layer as output and predict
            model_partial = Model(inputs=model_full.input, outputs=model_full.layers[out_layer].output)
            test_output = np.asarray(model_partial.predict(x_test))
            train_output = np.asarray(model_partial.predict(x_train))
            logger.info('test_output len -> ' + str(test_output.shape))
            logger.info('train_output len -> ' + str(train_output.shape))

            embeddings, speakers, num_embeddings = generate_embeddings(train_output, test_output, speakers_train,
                                                                       speakers_test, vector_size)

            # Fill the embeddings and speakers into the arrays
            set_of_embeddings.append(embeddings)
            set_of_speakers.append(speakers)
            speaker_numbers.append(num_embeddings)

            # Calculate the time per utterance
            time = TimeCalculator.calc_time_long_short_utterances(speakers_train, speakers_test, seg_size)
            set_of_total_times.append(time)

        logger.info('Pairwise_lstm test done.')
        return checkpoints, set_of_embeddings, set_of_speakers, speaker_numbers, set_of_total_times


def prepare_data(X,y, segment_size):
    x, speakers = generate_test_data(X, y, segment_size)

    # Reshape test data because it is an lstm
    return x.reshape(x.shape[0], x.shape[3], x.shape[2]), speakers
