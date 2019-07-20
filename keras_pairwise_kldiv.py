from common.utils.paths import *
from common.utils.load_config import *
from networks.pairwise_kldiv.keras_network_training.clustering_network import create_and_train
from networks.pairwise_kldiv.keras_network_training.network_factory import create_network_n_speakers

if __name__ == '__main__':
    config = load_config(None, join(get_common(), 'config.cfg'))
    net_file = get_experiment_nets("pairwise_kldiv_100.h5")
    train_file = get_speaker_pickle("speakers_100_50w_50m_not_reynolds_cluster")
    network = create_network_n_speakers(100, config)
    batch_size = config.getint('pairwise_kldiv', 'batch_size')
    epoch_batches = config.getint('pairwise_kldiv', 'epoch_batches')
    num_epochs = config.getint('pairwise_kldiv', 'num_epochs')

    create_and_train(num_epochs=num_epochs,
                     batch_size=batch_size,
                     network_params_file_out=net_file,
                     train_file=train_file,
                     epoch_batches=epoch_batches,
                     network=network)
