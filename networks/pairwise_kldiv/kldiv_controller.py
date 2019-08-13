from common.spectrogram.speaker_dev_selector import load_test_data
from common.network_controller import NetworkController
from common.utils.ShortUtteranceConverter import create_data_lists
from common.utils.logger import *
from common.utils.paths import get_speaker_pickle, get_experiment_nets
from networks.pairwise_kldiv.core.create_embeddings import create_embeddings

from networks.pairwise_kldiv.core.train_network import train_network
from networks.pairwise_kldiv.core.network_factory import create_network_n_speakers
from common.training.data_gen import DataGenerator


class KLDivController(NetworkController):
    def __init__(self, config, dev):
        super().__init__("pairwise_kldiv", config, dev)
        self.checkpoints = ["pairwise_kldiv_100.h5"]
        self.logger = get_logger('kldiv', logging.INFO)
        self.dg = DataGenerator(config.getint('pairwise_kldiv', 'seg_size'),
                                config.getint('pairwise_kldiv', 'spectrogram_height'))

    def train_network(self):
        # Get settings
        n_epochs = self.config.getint('pairwise_kldiv', 'n_epochs')
        batch_size = self.config.getint('pairwise_kldiv', 'batch_size')
        epoch_batches = self.config.getint('pairwise_kldiv', 'epoch_batches')
        train_filename = self.config.get('train', 'pickle')
        n_speakers = self.config.getint('train', 'n_speakers')

        # Create network, load path to input and output file
        network = create_network_n_speakers(n_speakers, self.config)
        train_file = get_speaker_pickle(train_filename)
        net_file = get_experiment_nets(self.checkpoints[0])

        train_network(network=network,
                      train_file=train_file,
                      network_file_out=net_file,
                      data_generator=self.dg,
                      num_epochs=n_epochs,
                      batch_size=batch_size,
                      epoch_batches=epoch_batches)

    def get_embeddings(self):
        # Get settings
        short_utterance = self.config.getboolean('test', 'short_utterances')
        out_layer = self.config.getint('pairwise_kldiv', 'out_layer')
        seg_size = self.config.getint('pairwise_kldiv', 'seg_size')
        self.logger.info('Run pairwise_kldiv')

        # Load and prepare test data
        X_long, y_long, s_list_long = load_test_data(self.get_validation_train_data())
        X_short, y_short, s_list_short = load_test_data(self.get_validation_test_data())
        X_long, y_long = self._prepare_data(X_long, y_long)
        X_short, y_short = self._prepare_data(X_short, y_short)

        x_list, y_list, _ = create_data_lists(short_utterance, X_long, X_short,
                                              y_long, y_short, s_list_long, s_list_short)

        embeddings_data =  create_embeddings(self.config, self.checkpoints, x_list, y_list, out_layer, seg_size)
        return embeddings_data

    def _prepare_data(self, X, y):
        x, speakers = self.dg.generate_test_data(X, y)

        # Reshape test data
        return x.reshape(x.shape[0], x.shape[1], x.shape[3], x.shape[2]), speakers
