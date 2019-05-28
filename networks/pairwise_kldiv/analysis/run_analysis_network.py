from theano import tensor as Tensor

from networks.pairwise_kldiv.network_training import clustering_network, clustering_network_foreign_conv, \
    network_factory


def run_analysis_network(network_file, X_input, y_input, speaker_list, conv_network_file=None, save_data_file=None):
    # Load eventual CNN output
    if conv_network_file is None:
        get_conv_output = None
    else:
        input_var = Tensor.tensor4('x')
        get_conv_output = clustering_network_foreign_conv.prepare(conv_network_file, input_var)

    # Generate output train data
    X_output, y_output, speaker_names = clustering_network.generate_output(X_input, y_input, speaker_list, network_params_file_in=network_file,
                                                                               output_file_out=None,
                                                                               network_fun=network_factory.create_network_470_speakers,
                                                                               get_conv_output=get_conv_output,
                                                                               output_layer=7)

    return X_output, y_output
