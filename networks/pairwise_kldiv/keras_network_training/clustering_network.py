import pickle

from common.utils.logger import *
from common.utils.paths import *
from common.utils.load_config import *
from networks.pairwise_kldiv.keras_network_training.network_factory import create_network_n_speakers
from networks.pairwise_lstm.core import data_gen as dg


config = load_config(None, join(get_common(), 'config.cfg'))
logger = get_logger("luvo", logging.INFO)


def create_and_train(num_epochs=1000, batch_size=100, epoch_batches=10,
                     network_params_file_out=None, train_file=None,
                     network=create_network_n_speakers(10, config)):
    seg_size = config.getint('pairwise_kldiv', 'seg_size')

    # load training data
    with open(train_file, 'rb') as f:
        X, y, speaker_names = pickle.load(f)

    #Train network
    train_gen = dg.batch_generator(X, y, batch_size=batch_size, segment_size=seg_size)
    logger.info("Fitting...")
    network.fit_generator(train_gen, steps_per_epoch=epoch_batches, epochs=num_epochs,
                                  verbose=2)

    #Save model
    logger.info("Saving model")
    network.save(network_params_file_out)
