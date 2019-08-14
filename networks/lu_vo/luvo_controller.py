"""
The controller to train and test the luvo network
"""

from common.network_controller import NetworkController
from common.utils.paths import *
from common.spectrogram.speaker_dev_selector import load_test_data
from networks.lu_vo.keras_network_training.spectrogram_cnn import SpectrogramCnn


class LuvoController(NetworkController):
    def __init__(self, name, config, dev):
        super().__init__(name, config, dev)
        self.checkpoint = self.name + ".pickle"
        self.cnn = SpectrogramCnn(self.name, get_experiment_nets(self.checkpoint), config)

    def train_network(self):
        train_file = self.config.get('train', 'pickle')
        self.cnn.create_and_train(train_file)

    def get_embeddings(self):
        X_train, y_train, s_list_train = load_test_data(self.get_validation_train_data())
        X_test, y_test, s_list_test = load_test_data(self.get_validation_test_data())

        embeddings, speakers, num_embeddings, time = self.cnn.create_embeddings(X_train, y_train, X_test, y_test)

        return [self.checkpoint], [embeddings], [speakers], [num_embeddings], [time]
