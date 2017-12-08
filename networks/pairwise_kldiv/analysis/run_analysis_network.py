from theano import tensor as Tensor

from networks.pairwise_kldiv.network_training import clustering_network, clustering_network_foreign_conv, \
    network_factory


def run_analysis_network(network_file, train_data_file, test_data_file, conv_network_file=None, save_data_file=None):
    # Load eventual CNN output
    if conv_network_file is None:
        get_conv_output = None
    else:
        input_var = Tensor.tensor4('x')
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

    return X_train, y_train, X_test, y_test
