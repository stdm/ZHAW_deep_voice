"""
The controller to train and test the luvo network
"""

from common.network_controller import NetworkController
from common.utils.logger import *
from common.utils.paths import *
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
        train_data = self.get_validation_train_data()
        test_data = self.get_validation_test_data()

        embeddings, speakers, num_embeddings = self.cnn.create_embeddings(train_data, test_data)

        return [self.checkpoint], [embeddings], [speakers], [num_embeddings]