"""
    This file takes the network and train/test files to evaluate the clusters and MR of this data.

    EDIT Heusser:
    - Rename from "main.py" to "analyse_network.py"
    - Create method analyse_network to make execution chainable

    Work of Lukic and Vogt, adapted by Heusser.
"""
from matplotlib import pyplot as plt
from theano import tensor as T

from common.utils.paths import *
from ..clustering import cluster
from ..network_training import clustering_network, clustering_network_foreign_conv, \
    network_factory


def analyse_network(network_file, train_data_file, test_data_file, conv_network_file=None, save_data_file=None):
    # Load eventual CNN output
    if conv_network_file is None:
        get_conv_output = None
    else:
        input_var = T.tensor4('x')
        get_conv_output = clustering_network_foreign_conv.prepare(conv_network_file, input_var)

    # Generate output train data
    X_train, y_train, speaker_names_train = clustering_network.generate_output(network_params_file_in=network_file,
                                                                               input_file=train_data_file,
                                                                               output_file_out=None,
                                                                               network_fun=network_factory.create_network_100_speakers,
                                                                               get_conv_output=get_conv_output,
                                                                               output_layer=7)

    # Generate output test data
    X_test, y_test, speaker_names_test = clustering_network.generate_output(network_params_file_in=network_file,
                                                                            input_file=test_data_file,
                                                                            output_file_out=None,
                                                                            network_fun=network_factory.create_network_100_speakers,
                                                                            get_conv_output=get_conv_output,
                                                                            output_layer=7)

    # Create MR
    X, y, num_speakers = cluster.generate_embedings(X_train, X_test, y_train, y_test, X_train.shape[1])
    MRs = cluster.calc_MR(X, y, len(set(y)), 'cosine')

    # Plot MR in clusters
    plt.plot(MRs, label='test', linewidth=2)
    plt.xlabel('Clusters')
    plt.ylabel('Misclassification Rate (MR)')
    plt.grid()
    plt.legend(loc='lower right', shadow=False)
    plt.ylim(0, 1)
    plt.show()

    # Save plot
    if save_data_file is not None:
        plt.savefig(save_data_file)


if __name__ == "__main__":
    NETWORK_FILE = get_experiment_nets('network_params_cluster_kl_100.pickle')
    TRAIN_DATA_FILE = get_speaker_pickle('train_data_40_clustering_vs_reynolds')
    TEST_DATA_FILE = get_speaker_pickle('test_data_40_clustering_vs_reynolds')

    analyse_network(network_file=NETWORK_FILE, train_data_file=TRAIN_DATA_FILE, test_data_file=TEST_DATA_FILE)
