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

        return [self.checkpoint], [embeddings], [speakers], [num_embeddings]