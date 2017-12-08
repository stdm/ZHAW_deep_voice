"""
    This file creates, trains and can safe a CNN and its parameters.

    Work of Lukic and Vogt, adapted by Heusser.
"""
from lasagne import layers
from lasagne import nonlinearities

from networks.pairwise_kldiv.core import settings
from .spectrogram_cnn import SpectrogramCnn


class SpectrogramCnn100(SpectrogramCnn):
    def create_paper(self, shape):
        paper = [
            # input layer
            (layers.InputLayer, {'shape': (None, shape, settings.FREQ_ELEMENTS, settings.ONE_SEC)}),

            # convolution layers 1
            (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (8, 1)}),
            (layers.BatchNormLayer, {}),
            (layers.MaxPool2DLayer, {'pool_size': 4, 'stride': 2}),

            # convolution layers 2
            (layers.Conv2DLayer, {'num_filters': 64, 'filter_size': (8, 1)}),
            (layers.BatchNormLayer, {}),
            (layers.MaxPool2DLayer, {'pool_size': 4, 'stride': 2}),

            # dense layer
            (layers.DenseLayer, {'num_units': 1000}),
            (layers.BatchNormLayer, {}),
            (layers.DropoutLayer, {}),
            (layers.DenseLayer, {'num_units': 500}),

            # output layer
            (layers.DenseLayer, {'num_units': 100, 'nonlinearity': nonlinearities.softmax})
        ]

        return paper
