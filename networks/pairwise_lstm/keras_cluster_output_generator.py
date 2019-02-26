"""
    This file allows you tu use a pre-existing trained network and run it on test data.
    It saves the output in a pickle so it can be evaluated further.

    Work of Gerber and Glinski.
"""
import pickle

import numpy as np
from keras.models import Model
from keras.models import load_model
from keras.models import model_from_json

from common.utils.paths import *
from .core import data_gen as dg
from .core import pairwise_kl_divergence as kld


def generate_cluster_output(network_name, test_data, output_file, one_file, write_to_file, is_LSTM, segment_size=15):
    with open(get_speaker_pickle(test_data), 'rb') as f:
        (X, y, s_list) = pickle.load(f)

    if one_file:
        model = load_model(
            get_experiment_nets(network_name + ".h5"),
            custom_objects={'pairwise_kl_divergence': kld.pairwise_kl_divergence}
        )

    else:
        json_file = open('../data/nets/cnn_speaker02.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("../data/nets/cnn_speaker.02h5")

    model.compile(loss=kld.pairwise_kl_divergence, optimizer='rmsprop', metrics=['accuracy', 'categorical_accuracy', ])

    print("Data extraction...")
    X_test, y_test = dg.generate_test_data(X, y, segment_size)
    print(X.shape)
    print(X_test.shape)

    print("Data extraction done!")
    if is_LSTM == True:
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[3], X_test.shape[2])

    print("Test output...")
    print(model.layers[1].output.get_shape())
    print(model.layers[3].output.get_shape())
    print(model.layers[4].output.get_shape())
    im_model = Model(inputs=model.input, outputs=model.layers[2].output)
    data_out = im_model.predict(X_test)
    da = np.asarray(data_out)

    with open(get_experiment_cluster(output_file), 'wb') as f:
        pickle.dump((da, y_test, s_list), f, -1)


if __name__ == "__main__":
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'test_clustering_10.pickle', 'test_cluster_out_10sp_500ms_256_400sp_kld', True, True, True)
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'train_clustering_10.pickle', 'train_cluster_out_10sp_500ms_256_400sp_kld', True, True, True)
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'test_clustering_40.pickle', 'test_cluster_out_40sp_500ms_256_400sp_kld', True, True, True)
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'train_clustering_40.pickle', 'train_cluster_out_40sp_500ms_256_400sp_kld', True, True, True)
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'test_clustering_80.pickle', 'test_cluster_out_80sp_500ms_256_400sp_kld', True, True, True)
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'train_clustering_80.pickle', 'train_cluster_out_80sp_500ms_256_400sp_kld', True, True, True)
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'test_test_clustering_100.pickle', 'test_cluster_out_100sp_500ms_256_400sp_kld', True, True, True)
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'train_test_clustering_100.pickle', 'train_cluster_out_100sp_500ms_256_400sp_kld', True, True, True)
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'test_test_clustering_120.pickle', 'test_cluster_out_120sp_500ms_256_400sp_kld', True, True, True)
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'train_test_clustering_120.pickle', 'train_cluster_out_120sp_500ms_256_400sp_kld', True, True, True)
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'test_test_clustering_20.pickle', 'test_cluster_out_20sp_500ms_256_400sp_kld', True, True, True)
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'train_test_clustering_20.pickle', 'train_cluster_out_20sp_500ms_256_400sp_kld', True, True, True)
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'test_test_clustering_60.pickle', 'test_cluster_out_60sp_500ms_256_400sp_kld', True, True, True)
    # generate_cluster_output('cluster_train_droput_500ms_256_100sp_kld', 'train_test_clustering_60.pickle', 'train_cluster_out_60sp_500ms_256_400sp_kld', True, True, True)
    #generate_cluster_output('20170425_lstm2_dense_kld_100batch', 'train_speakers_40_clustering_vs_reynolds',
    #                        '20170425_lstm2_dense_kld_100batch_train_cluster_40.pickle', True, True, True)
    #generate_cluster_output('20170425_lstm2_dense_kld_100batch', 'test_speakers_40_clustering_vs_reynolds',
    #                        '20170425_lstm2_dense_kld_100batch_test_cluster_40.pickel', True, True, True)
    #generate_cluster_output('20170425_lstm2_dense_kld_100batch', 'train_speakers_60_clustering',
    #                        '20170425_lstm2_dense_kld_100batch_train_cluster_60.pickel', True, True, True)
    #generate_cluster_output('20170425_lstm2_dense_kld_100batch', 'test_speakers_60_clustering',
    #                        '20170425_lstm2_dense_kld_100batch_test_cluster_60', True, True, True)
    #generate_cluster_output('20170425_lstm2_dense_kld_100batch', 'train_speakers_80_clustering',
    #                        '20170425_lstm2_dense_kld_100batch_train_cluster_80.pickel', True, True, True)
    generate_cluster_output('pairwise_lstm_100_best', 'speakers_40_clustering_vs_reynolds_cluster',
                            'pairwise_lstm_100_100batch_test_cluster_40.pickel', True, True, True)
    # generate_cluster_output('kldold_clust_lstm4', 'train_speakers_40_clustering_vs_reynolds.pickle', 'train_cluster_40_lstm4_layer_lstm4_out.pickel', True, True, True)
    # generate_cluster_output('kldold_clust_lstm4', 'test_speakers_40_clustering_vs_reynolds.pickle', 'test_cluster_40_lstm4_layer_lstm4_out.pickel', True, True, True)
    # generate_cluster_output('kldold_clust_lstm4_dense', 'train_speakers_40_clustering_vs_reynolds.pickle', 'train_cluster_40_lstm4d_layer_lstm4_out.pickel', True, True, True)
    # generate_cluster_output('kldold_clust_lstm4_dense', 'test_speakers_40_clustering_vs_reynolds.pickle', 'test_cluster_40_lstm4d_layer_lstm4_out.pickel', True, True, True)
