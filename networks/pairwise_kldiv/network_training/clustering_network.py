"""
    This file creates, trains and can safe a network and its parameters.

    EDIT Heusser:
    - Rename method main to create_and_train
    - move generate_cluster_data here

    Work of Lukic and Vogt.
"""
import pickle
import sys
import time

import lasagne
import numpy as np
import theano
import theano.tensor as T

import common.spectogram.speaker_train_splitter as sts
from common.spectogram.spectrogram_extractor import extract_spectrogram
from common.utils import pickler
from common.utils.paths import *
from . import network_factory as nf
from .objectives_clustering import create_loss_functions_kl_div
from ..core import analytics, settings
from ..core.batch_iterators import SpectTrainBatchIterator, SpectValidBatchIterator


def create_and_train(num_epochs=1000, batch_size=100, epoch_batches=10, network_params_file_in=None,
                     network_params_file_out=None,
                     train_file=None,
                     network_fun=nf.create_network_10_speakers, with_validation=True):
    # load training data
    with open(train_file, 'rb') as f:
        (X, y, speaker_names) = pickle.load(f)

    # create symbolic theano variables
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    margin = T.scalar('margin')

    # create network
    network = network_fun(input_var)
    if network_params_file_in is not None:
        all_param_values = pickler.load(network_params_file_in)
        lasagne.layers.set_all_param_values(network, all_param_values)

    train_fn, val_fn = create_loss_functions_kl_div(input_var, network, target_var, margin)

    # start training
    if not with_validation:
        val_fn = None

    # Train network
    train(X, y, num_epochs, train_fn, val_fn, SpectTrainBatchIterator(batch_size, epoch_batches, 1, 1),
          SpectValidBatchIterator(batch_size, epoch_batches))

    # Save if
    if network_params_file_out is not None:
        pickler.save(lasagne.layers.get_all_param_values(network), network_params_file_out)


def train(X, y, num_epochs, train_fn, val_fn=None, train_iterator=None, validation_iterator=None):
    if train_iterator is None:
        train_iterator = SpectTrainBatchIterator(100, 10)
    if validation_iterator is None:
        validation_iterator = SpectValidBatchIterator(100, 10)
    if val_fn is not None:
        train_splitter = sts.SpeakerTrainSplit(eval_size=0.25, sentences=8)
        X_train, X_valid, y_train, y_valid = train_splitter(X, y)
    else:
        X_train = X
        y_train = y

    margin = 2

    print("Start training")
    confusion = analytics.ConfusionMatrix(len(set(y)))

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training
        train_err = 0
        train_batches = 0
        for Xb, yb in train_iterator.iterate(X_train, y_train):
            train_err += train_fn(Xb, yb, margin)
            train_batches += 1

        # Validation
        if val_fn is not None:
            val_err = 0
            val_batches = 0
            for Xb, yb in validation_iterator.iterate(X_valid, y_valid):
                err, class_predictions = val_fn(Xb, yb, margin)
                val_err += err
                confusion.add_predictions(yb, class_predictions)
                val_batches += 1

            accs = confusion.calculate_accuracies()
            confusion.mat.fill(0)

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

        if val_fn is not None:
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                accs.sum() / len(accs) * 100))

        sys.stdout.flush()


def generate_output(network_params_file_in=None, input_file=None, output_file_out=None, network_fun=None,
                    get_conv_output=None, output_layer=None, overlapping=False):
    with open(
            input_file,
            'rb') as f:
        (X, y, speaker_names) = pickle.load(f)

    X_cluster, y_cluster = generate_cluster_data(X, y, overlapping=overlapping)

    input_var = T.tensor4('inputs')
    network = network_fun(input_var)
    if network_params_file_in is not None:
        all_param_values = pickler.load(network_params_file_in)
        lasagne.layers.set_all_param_values(network, all_param_values)

    if output_layer is not None:
        network = lasagne.layers.get_all_layers(network)[output_layer]
    prediction = lasagne.layers.get_output(network, deterministic=True)
    predict = theano.function([input_var], prediction)

    data_len = len(X_cluster)
    if get_conv_output is not None:
        X_conv = np.zeros((data_len, 1, 1, 1000), dtype=np.float32)
    output = np.zeros((data_len, network.num_units), dtype=np.float32)
    step = 100
    for i in range(0, data_len, step):
        next_step = data_len - i
        if next_step > step:
            next_step = step
        if get_conv_output is None:
            new_output = predict(X_cluster[i:i + next_step])
        else:
            X_conv[i:i + next_step, 0, 0] = get_conv_output(X_cluster[i:i + next_step])
            new_output = predict(X_conv[i:i + next_step])
        output[i:i + step] = new_output
    if output_file_out is None:
        return output, y_cluster, speaker_names
    else:
        with open(output_file_out, 'wb') as f:
            pickle.dump((output, y_cluster, speaker_names), f, -1)


def generate_cluster_data(X, y, overlapping=False):
    X_cluster = np.zeros((10000, 1, settings.FREQ_ELEMENTS, settings.ONE_SEC), dtype=np.float32)
    y_cluster = []

    step = settings.ONE_SEC
    if overlapping:
        step = settings.ONE_SEC / 2
    pos = 0
    for i in range(len(X)):
        spect = extract_spectrogram(X[i, 0], settings.ONE_SEC, settings.FREQ_ELEMENTS)

        for j in range(int(spect.shape[1] / step)):
            y_cluster.append(y[i])
            seg_idx = j * step
            try:
                X_cluster[pos, 0] = spect[:, seg_idx:seg_idx + settings.ONE_SEC]
            except ValueError:
                # if the last segment doesn't match ignore it
                pass
            pos += 1

    return X_cluster[0:len(y_cluster)], np.asarray(y_cluster, dtype=np.int32)


if __name__ == "__main__":
    net_file = get_experiment_nets("fs16_ba_LV_KL_590.pickle")
    train_file = get_speaker_pickle("train_speakers_590_clustering_without_raynolds")

    create_and_train(network_params_file_in=None,
                     network_params_file_out=net_file,
                     train_file=train_file,
                     network_fun=nf.create_network_590_speakers, with_validation=False)

    # generate_output(
    #    network_params_file_in='../../data/experiments/clusteringKL/networks/10_speakers_network_params.pickle',
    #    input_file='../../data/training/speaker_pickles/test_data_10_clustering_vs_reynolds.pickle',
    #    output_file_out='../../data/experiments/clusteringKL/outputs/test_output_10_reynolds.pickle',
    #    network_fun=nf.create_network_10_speakers)
