"""
The controller to train and test the pairwise_lstm network
"""

import numpy as np
import random
from keras.models import Model
from keras.models import load_model

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils.logger import *
from common.utils.paths import *
from common.utils.pickler import load
from .bilstm_2layer_dropout_plus_2dense import bilstm_2layer_dropout
from .core.data_gen import generate_test_data
from .core.pairwise_kl_divergence import pairwise_kl_divergence
from common.utils.load_config import *
from common.spectogram.speaker_dev_selector import get_sentences_for_speaker_index

class LSTMController(NetworkController):
    def __init__(self, out_layer, seg_size, vec_size):
        super().__init__("pairwise_lstm")
        self.network_file = self.name + "_100"
        self.out_layer = out_layer
        self.seg_size = seg_size
        self.vec_size = vec_size

    def train_network(self):
        config = load_config(None, join(get_common(), 'config.cfg'))
        bilstm_2layer_dropout(
            self.network_file,
            config.get('train', 'pickle'),
            config.getint('pairwise_lstm', 'n_hidden1'),
            config.getint('pairwise_lstm', 'n_hidden2'),
            config.getint('pairwise_lstm', 'n_classes'),
            config.getint('pairwise_lstm', 'n_10_batches'),
            segment_size=self.seg_size
        )

    def get_embeddings(self, out_layer, seg_size, vec_size):
        logger = get_logger('lstm', logging.INFO)
        logger.info('Run pairwise_lstm test')
        logger.info('out_layer -> ' + str(self.out_layer))
        logger.info('seg_size -> ' + str(self.seg_size))
        logger.info('vec_size -> ' + str(self.vec_size))

        # Load and prepare train/test data
        if self.dev_mode:
            x_train, speakers_train, x_test, speakers_test = \
                load_dev_test_data(self.get_validation_train_data(), self.get_validation_test_data(), self.seg_size, self.val_data_size)
        else:
            x_test, speakers_test = load_and_prepare_data(self.get_validation_test_data(), self.seg_size)
            x_train, speakers_train = load_and_prepare_data(self.get_validation_train_data(), self.seg_size)

        # Prepare return values
        set_of_embeddings = []
        set_of_speakers = []
        speaker_numbers = []
        checkpoints = list_all_files(get_experiment_nets(), "*pairwise_lstm*.h5")

        # Values out of the loop
        metrics = ['accuracy', 'categorical_accuracy', ]
        loss = pairwise_kl_divergence
        custom_objects = {'pairwise_kl_divergence': pairwise_kl_divergence}
        optimizer = 'rmsprop'
        vector_size = self.vec_size #256 * 2

        # Fill return values
        for checkpoint in checkpoints:
            logger.info('Running checkpoint: ' + checkpoint)
            # Load and compile the trained network
            network_file = get_experiment_nets(checkpoint)
            model_full = load_model(network_file, custom_objects=custom_objects)
            model_full.compile(loss=loss, optimizer=optimizer, metrics=metrics)

            # Get a Model with the embedding layer as output and predict
            model_partial = Model(inputs=model_full.input, outputs=model_full.layers[self.out_layer].output)
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

        logger.info('Pairwise_lstm test done.')
        return checkpoints, set_of_embeddings, set_of_speakers, speaker_numbers

'''
This method randomly chooses a number of speakers out of a test set.
As of now, this method should only be used for dev-tests to prevent overfitting during development. (see BA_2019 for details)
'''
def load_dev_test_data(long_utterance_path, short_utterance_path, segment_size, number_speakers):
    x_train, y_train, s_list_train = load(long_utterance_path)
    x_test, y_test, s_list_test = load(short_utterance_path)

    if number_speakers == 80:
        x_test, speakers_test = generate_test_data(x_test[:(number_speakers*2)], y_test[:(number_speakers*2)], segment_size)
        x_train, speakers_train = generate_test_data(x_train[:number_speakers*8], y_train[:(number_speakers*2)], segment_size)
        return x_train.reshape(x_train.shape[0], x_train.shape[3], x_train.shape[2]), speakers_train, x_test.reshape(x_test.shape[0], x_test.shape[3], x_test.shape[2]), speakers_test
    else:
        x_train_sampled = np.zeros(x_train.shape)
        y_train_sampled = np.zeros(y_train.shape)
        x_test_sampled = np.zeros(x_test.shape)
        y_test_sampled = np.zeros(y_test.shape)
        for i in range(number_speakers):
            index = random.randrange(80-i)

            x_sampled, y_sampled, x_train, y_train = get_sentences_for_speaker_index(x_train, y_train, index, i, 8)
            x_train_sampled[i*8:i*8+8] = x_sampled
            y_train_sampled[i*8:i*8+8] = y_sampled
            x_sampled, y_sampled, x_test, y_test = get_sentences_for_speaker_index(x_test, y_test, index, i, 2)
            x_test_sampled[i*2:i*2+2] = x_sampled
            y_test_sampled[i * 2:i * 2 + 2] = y_sampled

        x_test, speakers_test = generate_test_data(x_test[:(number_speakers*2)], y_test[:(number_speakers*2)], segment_size)
        x_train, speakers_train = generate_test_data(x_train[:(number_speakers * 2)], y_train[:(number_speakers * 2)], segment_size)

        return x_train.reshape(x_train.shape[0], x_train.shape[3], x_train.shape[2]), speakers_train, x_test.reshape(
            x_test.shape[0], x_test.shape[3], x_test.shape[2]), speakers_test



def load_and_prepare_data(data_path, segment_size):
    # Load and generate test data
    x, y, s_list = load(data_path)
    x, speakers = generate_test_data(x, y, segment_size)

    # Reshape test data because it is an lstm
    return x.reshape(x.shape[0], x.shape[3], x.shape[2]), speakers
