"""
    This file creates, trains and  can safe a network and its parameters.

    Instead of creating a network with convolutions, it takes a pre-trained CNN and uses them instead.

    Work of Lukic and Vogt.
"""

import pickle

import theano
import theano.tensor as T
from lasagne import layers

from common.utils import pickler
from common.utils.paths import *
from . import clustering_network, network_factory
from .objectives_clustering import create_loss_functions_kl_div
from ..core.batch_iterators import SpectWithSeparateConvTrainBatchIterator, \
    SpectWithSeparateConvValidBatchIterator


def create_and_train(network_file, train_file, out_file):
    print("Loading static convolution...")
    x = T.tensor4('x')
    get_conv_output = prepare(network_file, x)
    print("Static convolution loaded!")

    print("Create vanilla network...")
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    margin = T.scalar('margin')
    network = network_factory.create_network_KL_clustering_no_convolution(input_var, input_size=1000, output_size=100)
    train_fn, val_fn = create_loss_functions_kl_div(input_var, network, target_var, margin)
    train_batch_iterator = SpectWithSeparateConvTrainBatchIterator(batchsize=100, batches_per_epoch=10, input_dim=1000,
                                                                   get_conv_output=get_conv_output)
    valid_batch_iterator = SpectWithSeparateConvValidBatchIterator(batchsize=100, batches_per_epoch=10, input_dim=1000,
                                                                   get_conv_output=get_conv_output)
    print("Vanilla network created!")

    with open(train_file, 'rb') as f:
        (X, y, speaker_names) = pickle.load(f)
    clustering_network.train(X, y, num_epochs=1000, train_fn=train_fn, val_fn=val_fn,
                             train_iterator=train_batch_iterator, validation_iterator=valid_batch_iterator)

    pickler.save(layers.get_all_param_values(network), out_file)


def prepare(network_file, input_var):
    net = pickler.load(network_file)
    conv_network = net.get_all_layers()[5]
    y = layers.get_output(conv_network, input_var)
    return theano.function([input_var], y)


if __name__ == '__main__':
    network_file = get_experiment_nets('net_100_81_not_reynolds.pickle')
    train_file = get_speaker_pickle('train_data_100_50w_50m_not_reynolds')
    out_file = get_experiment_nets('160419_net_static_conv_kldiv_100_01.pickle')

    create_and_train(network_file, train_file, out_file)
