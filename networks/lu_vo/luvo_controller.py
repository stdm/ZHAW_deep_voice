"""
The controller to train and test the luvo network
"""

from common.network_controller import NetworkController
from common.utils.logger import *
from common.utils.paths import *
from common.spectrogram.speaker_dev_selector import load_test_data, load_dev_test_data
from .network_training.spectrogram_cnn_590 import SpectrogramCnn590
from common.utils.load_config import *


class LuvoController(NetworkController):
    def __init__(self):
        super().__init__("luvo")
        self.checkpoint = self.name + ".pickle"
        self.logger = get_logger(self.name, logging.INFO)
        self.config = load_config(None, join(get_common(), 'config.cfg'))
        self.cnn = SpectrogramCnn590(get_experiment_nets(self.checkpoint), self.config)

    def train_network(self):
        train_file = get_speaker_pickle(self.config.get('train', 'pickle'))
        self.cnn.create_and_train(train_file)

    def get_embeddings(self):
        X_train, y_train, s_list_train = load_test_data(self.get_validation_train_data())
        X_test, y_test, s_list_test = load_test_data(self.get_validation_test_data())

        embeddings, speakers, num_embeddings, time = self.cnn.create_embeddings(X_train, y_train, X_test, y_test)

        return [self.checkpoint], [embeddings], [speakers], [num_embeddings], [time]