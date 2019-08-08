import pickle

from common.utils.logger import *


logger = get_logger("kldiv", logging.INFO)


def train_network(network, train_file, network_file_out, data_generator,
                  num_epochs=1000, batch_size=100, epoch_batches=10):
    # Load training data
    with open(train_file, 'rb') as f:
        X, y, speaker_names = pickle.load(f)

    # Train network
    train_gen = data_generator.batch_generator_cnn(X, y, batch_size=batch_size)
    logger.info("Fitting...")
    network.fit_generator(train_gen, steps_per_epoch=epoch_batches, epochs=num_epochs, verbose=2)

    # Save model
    logger.info("Saving model")
    network.save(network_file_out)
