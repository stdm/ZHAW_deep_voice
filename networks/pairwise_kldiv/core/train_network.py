import pickle

from common.utils.logger import *
from networks.pairwise_lstm.core import data_gen as dg


logger = get_logger("kldiv", logging.INFO)


def train_network(network, train_file, network_file_out,
                  num_epochs=1000, batch_size=100, epoch_batches=10, seg_size=100):
    # Load training data
    with open(train_file, 'rb') as f:
        X, y, speaker_names = pickle.load(f)

    # Train network
    train_gen = dg.batch_generator(X, y, batch_size=batch_size, segment_size=seg_size)
    logger.info("Fitting...")
    network.fit_generator(train_gen, steps_per_epoch=epoch_batches, epochs=num_epochs, verbose=2)

    # Save model
    logger.info("Saving model")
    network.save(network_file_out)
