import pickle

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras.models import Model
from keras.models import load_model
from keras.models import model_from_json
from keras.utils import np_utils
from sklearn.metrics import accuracy_score

from common.utils.paths import *
from ..core import data_gen as dg

'''
calculates Accuracies for the provided Networks as well as k nearest for k = {2, 3, 5, 10}

network_name: network_name of the network to load from settings.NET_PATH
test_data: test_data to load from settings.DATA_PATH
one_file: Boolean expectet true if the network is saved in H5 format.
write_to_file: True if Results should be logged to test_scores.txt in the settings.LOG_PATH
is_LSTM: True if the tested Network requires Input shape for LSTM
segment_size: Segement Size needs to be the same as was used during training.
'''


def calculate_test_accuracies(network_name, test_data, one_file, write_to_file, is_LSTM, segment_size=15):
    with open(get_speaker_pickle(test_data), 'rb') as f:
        (X, y, s_list) = pickle.load(f)

    if one_file:
        model = load_model(get_experiment_nets(network_name + '.h5'))
    else:
        json_file = open(get_experiment_nets('cnn_speaker02.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(get_experiment_nets('cnn_speaker.02h5'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'categorical_accuracy', ])

    print("Data extraction...")
    X_test, y_test = dg.generate_test_data(X, y, segment_size)
    n_classes = np.amax(y_test) + 1

    print("Data extraction done!")
    if is_LSTM:
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[3], X_test.shape[2])

    print("Test output...")
    im_model = Model(input=model.input, output=model.layers[2].output)
    data_out = im_model.predict(X_test, batch_size=128)
    print(data_out)
    da = np.asarray(data_out)
    np.savetxt("foo.csv", da, delimiter=",")
    with open(test_data + "cluster_out_01", 'wb') as f:
        pickle.dump((da, y, s_list), f, -1)

    output = model.predict(X_test, batch_size=128, verbose=1)
    y_t = np_utils.to_categorical(y_test, n_classes)
    eva = model.evaluate(X_test, y_t, batch_size=128, verbose=2)
    k_nearest2 = K.eval(metrics.top_k_categorical_accuracy(tf.stack(y_t), tf.stack(output), k=2))
    k_nearest3 = K.eval(metrics.top_k_categorical_accuracy(tf.stack(y_t), tf.stack(output), k=3))
    k_nearest5 = K.eval(metrics.top_k_categorical_accuracy(tf.stack(y_t), tf.stack(output), k=5))
    k_nearest10 = K.eval(metrics.top_k_categorical_accuracy(tf.stack(y_t), tf.stack(output), k=10))

    print(output.shape)
    output_sum = np.zeros((n_classes, n_classes))
    output_geom = np.zeros((n_classes, n_classes))
    y_pred_max = np.zeros(n_classes)
    y_pred_median = np.zeros(n_classes)
    for i in range(n_classes):
        indices = np.where(y_test == i)[0]
        speaker_output = np.take(output, indices, axis=0)
        max_val = 0
        for o in speaker_output:
            output_sum[i] = np.add(output_sum[i], o)
            output_geom[i] = np.multiply(output_geom[i], o)

            if np.max(o) > max_val:
                max_val = np.max(o)
                y_pred_max[i] = np.argmax(o)
        output_geom[i] = np.power(output_geom[i], 1 / len(speaker_output))

    y_pred_mean = np.zeros(n_classes)
    y_pred_geom = np.zeros(n_classes)
    for i in range(len(output_sum)):
        y_pred_mean[i] = np.argmax(output_sum[i])
        y_pred_geom[i] = np.argmax(output_sum[i])

    y_correct = np.arange(n_classes)

    print("geometric wrong")
    for j in range(len(y_correct)):
        if y_correct[j] != y_pred_geom[j]:
            print("Speaker: " + str(y_correct[j]) + ", Pred: " + str(y_pred_geom[j]))
            ind = np.argpartition(output_sum[j], -5)[-5:]
            print(np.argmax(output_sum[j]))
            print(ind[np.argsort(output_sum[j][ind])])

    print("mean wrong")
    for j in range(len(y_correct)):
        if y_correct[j] != y_pred_mean[j]:
            print("Speaker: " + str(y_correct[j]) + ", Pred: " + str(y_pred_mean[j]))
            ind = np.argpartition(output_sum[j], -5)[-5:]
            print(np.argmax(output_sum[j]))
            print(ind[np.argsort(output_sum[j][ind])])

    print(model.metrics_names)
    print(eva)
    print("Acc: %.4f" % eva[2])
    print("k2: %.4f" % k_nearest2)
    print("k3: %.4f" % k_nearest3)
    print("k5: %.4f" % k_nearest5)
    print("k10: %.4f" % k_nearest10)
    print("Accuracy (Max.): %.4f" % accuracy_score(y_correct, y_pred_max))
    print("Accuracy (Mean): %.4f" % accuracy_score(y_correct, y_pred_mean))
    print("Accuracy (Geom): %.4f" % accuracy_score(y_correct, y_pred_geom))
    if write_to_file == True:
        with open(get_experiment_logs('test_scores.txt'), 'ab') as f:
            f.write('---------- ' + network_name + '---------------\n')
            f.write("Accuracy: %.4f \n" % eva[2])
            f.write("Accuracy (Max.): %.4f \n" % accuracy_score(y_correct, y_pred_max))
            f.write("Accuracy (Mean): %.4f \n" % accuracy_score(y_correct, y_pred_mean))
            f.write("Accuracy (Geom): %.4f \n" % accuracy_score(y_correct, y_pred_geom))
            f.write("K2: {:.4}, K5: {:.4}, K10: {:.4} \n\n".format(k_nearest2, k_nearest5, k_nearest10))
