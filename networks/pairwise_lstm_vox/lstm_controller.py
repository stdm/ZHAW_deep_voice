"""
The controller to train and test the pairwise_lstm network
"""

import numpy as np
from keras.models import Model
from keras.models import load_model
from time import gmtime, strftime

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils.logger import *
from common.utils.paths import *
from common.utils.pickler import load_speaker_pickle_or_h5
from .bilstm_2layer_dropout_plus_2dense import bilstm_2layer_dropout
from .core.data_gen import generate_test_data
from .core.pairwise_kl_divergence import pairwise_kl_divergence


class LSTMVOX2Controller(NetworkController):
    def __init__(self, out_layer, seg_size, vec_size, 
                 active_learning_rounds, epochs, epochs_before_active_learning, 
                 dense_factor):
        super().__init__("pairwise_lstm_vox2", "vox2_speakers_120_test_cluster")
        self.out_layer = out_layer
        self.seg_size = seg_size
        self.vec_size = vec_size
        self.active_learning_rounds = active_learning_rounds
        self.epochs = epochs
        self.epochs_before_active_learning = epochs_before_active_learning
        self.dense_factor = dense_factor

        self.train_data = "vox2_speakers_5994_dev_cluster"
        self.val_data = "vox2_speakers_120_test_cluster"

    
    def get_validation_train_data(self):
        return get_speaker_pickle(self.train_data, ".h5")

    def get_validation_test_data(self):
        return get_speaker_pickle(self.val_data, ".h5")

    def get_network_name(self):
        return "{}__{}__{}__{}__{}__{}".format(
            self.name, 
            self.train_data, 
            self.dense_factor, 
            self.epochs, 
            self.epochs_before_active_learning, 
            self.active_learning_rounds
        )

    def train_network(self):
        bilstm_2layer_dropout(
            self.get_network_name(), 
            self.train_data, # _train suffix for train/test split, _cluster otherwise
            # 'vox2_speakers_120_test_cluster', # _train suffix for train/test split, _cluster otherwise
            # 'vox2_speakers_10_test_cluster', # _train suffix for train/test split, _cluster otherwise
            n_hidden1=256, 
            n_hidden2=256, 
            dense_factor=self.dense_factor, 
            epochs=self.epochs,
            epochs_before_active_learning=self.epochs_before_active_learning,
            active_learning_rounds=self.active_learning_rounds,
            segment_size=self.seg_size
        )

    def get_embeddings(self, out_layer, seg_size, vec_size):
        logger = get_logger('lstm', logging.INFO)
        logger.info('Run pairwise_lstm test')
        logger.info('out_layer -> ' + str(self.out_layer))
        logger.info('seg_size -> ' + str(self.seg_size))
        logger.info('vec_size -> ' + str(self.vec_size))

        # Load and prepare train/test data
        x_test, speakers_test = load_and_prepare_data(self.get_validation_test_data(), self.seg_size)
        x_train, speakers_train = load_and_prepare_data(self.get_validation_train_data(), self.seg_size)

        # Prepare return values
        set_of_embeddings = []
        set_of_speakers = []
        speaker_numbers = []
        checkpoints = list_all_files(get_experiment_nets(), self.get_network_name() + "*.h5")

        # Values out of the loop
        metrics = ['accuracy', 'categorical_accuracy', ]
        loss = pairwise_kl_divergence
        custom_objects = { 'pairwise_kl_divergence': pairwise_kl_divergence }
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

            embeddings, speakers, num_embeddings = generate_embeddings(
                train_output, test_output, speakers_train,
                speakers_test, vector_size
            )

            # Fill the embeddings and speakers into the arrays
            set_of_embeddings.append(embeddings)
            set_of_speakers.append(speakers)
            speaker_numbers.append(num_embeddings)

        logger.info('Pairwise_lstm test done.')
        return checkpoints, set_of_embeddings, set_of_speakers, speaker_numbers


def load_and_prepare_data(data_path, segment_size):
    # Load and generate test data
    (X, y, _) = load_speaker_pickle_or_h5(data_path)
    X, speakers = generate_test_data(X, y, segment_size)

    # Reshape test data because it is an lstm
    return X.reshape(X.shape[0], X.shape[3], X.shape[2]), speakers
