"""
The controller to train and test the pairwise_kldiv network
"""

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils import TimeCalculator
from common.utils.ShortUtteranceConverter import create_data_lists
from common.utils.logger import *
from .analysis.run_analysis_network import run_analysis_network
from .network_training.clustering_network import create_and_train
from .network_training.network_factory import *
from common.spectrogram.speaker_dev_selector import load_test_data


class KLDivController(NetworkController):
    def __init__(self, config):
        super().__init__("pairwise_kldiv", config)
        self.checkpoints = ["pairwise_kldiv_100.pickle"]

    def train_network(self):
        net_file = get_experiment_nets(self.checkpoints[0])
        train_file = get_speaker_pickle(config.get('train', 'pickle'),)

        create_and_train(num_epochs=config.getint('pairwise_kldiv', 'num_epochs'),
                         batch_size=config.getint('pairwise_kldiv', 'batch_size'),
                         network_params_file_in=None,
                         network_params_file_out=net_file,
                         train_file=train_file,
                         epoch_batches=config.getint('pairwise_kldiv', 'epoch_batches'),
                         network_fun=create_network_100_speakers, with_validation=False)

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
