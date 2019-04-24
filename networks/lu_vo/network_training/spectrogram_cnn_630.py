"""
    See spectrogram_cnn for usage.

    Implementation for 630 speakers.

    Work of Lukic and Vogt, adapted by Heusser
"""

from lasagne import layers
from lasagne import nonlinearities

from .spectrogram_cnn import SpectrogramCnn


class SpectrogramCnn630(SpectrogramCnn):
    def __init__(self, net_path, config):
        super().__init__(net_path)
        self.config = config

    def create_paper(self, shape):
        paper = [
            # input layer
            (layers.InputLayer, {'shape': (None, shape, self.config.getint('luvo', 'spectogram_height'),
                                           self.config.getint('luvo', 'seg_size'))}),

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
