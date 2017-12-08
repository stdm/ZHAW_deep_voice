"""
    See spectrogram_cnn for usage.

    Implementation for 630 speakers.

    Work of Lukic and Vogt, adapted by Heusser
"""

from lasagne import layers
from lasagne import nonlinearities

from networks.pairwise_kldiv.core import settings
from .spectrogram_cnn import SpectrogramCnn


class SpectrogramCnn630(SpectrogramCnn):
    def create_paper(self, shape):
        paper = [
            # input layer
            (layers.InputLayer, {'shape': (None, shape, settings.FREQ_ELEMENTS, settings.ONE_SEC)}),

            # convolution layers 1
            (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (4, 4)}),
            (layers.MaxPool2DLayer, {'pool_size': (4, 4), 'stride': (2, 2)}),

            # convolution layers 2
            (layers.Conv2DLayer, {'num_filters': 64, 'filter_size': (4, 4)}),
            (layers.MaxPool2DLayer, {'pool_size': (4, 4), 'stride': (2, 2)}),

            # dense layer
            (layers.DenseLayer, {'num_units': 6300}),
            (layers.DropoutLayer, {}),
            (layers.DenseLayer, {'num_units': 3150}),

            # output layer
            (layers.DenseLayer, {'num_units': 630, 'nonlinearity': nonlinearities.softmax})
        ]

        return paper
