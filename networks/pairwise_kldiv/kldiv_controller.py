"""
The controller to train and test the pairwise_kldiv network
"""

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils import TimeCalculator
from common.utils.logger import *
from common.utils.paths import *
from .analysis.run_analysis_network import run_analysis_network
from .network_training.clustering_network import create_and_train
from .network_training.network_factory import *
from common.utils.load_config import *
from common.spectogram.speaker_dev_selector import load_test_data, load_dev_test_data


class KLDivController(NetworkController):
    def __init__(self):
        super().__init__("pairwise_kldiv")
        self.checkpoints = ["pairwise_kldiv_100.pickle"]

    def train_network(self):
        config = load_config(None, join(get_common(), 'config.cfg'))
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
        logger = get_logger('kldiv', logging.INFO)
        logger.info('Run pairwise_kldiv')
        checkpoints = self.checkpoints

        X_train, y_train, s_list_train = load_test_data(self.get_validation_train_data())
        X_test, y_test, s_list_test = load_test_data(self.get_validation_test_data())

        # Prepare return value
        set_of_embeddings = []
        set_of_speakers = []
        set_of_num_embeddings = []
        set_of_total_times = []

        for checkpoint in checkpoints:
            logger.info('Run checkpoint: ' + checkpoint)
            network_file = get_experiment_nets(checkpoint)
            X_train, y_train, \
            X_test, y_test = run_analysis_network(network_file, X_train, y_train, s_list_train, X_test, y_test, s_list_test)
            embeddings, speakers, num_embeddings = generate_embeddings(X_train, X_test, y_train, y_test,
                                                                       X_train.shape[1])
            # Fill return values
            set_of_embeddings.append(embeddings)
            set_of_speakers.append(speakers)
            set_of_num_embeddings.append(num_embeddings)

            # Calculate the time per utterance
            time = TimeCalculator.calc_time_long_short_utterances(y_train, y_test, settings.ONE_SEC)
            set_of_total_times.append(time)

        return checkpoints, set_of_embeddings, set_of_speakers, set_of_num_embeddings, set_of_total_times
