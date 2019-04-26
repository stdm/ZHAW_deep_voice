"""
    See spectrogram_cnn for usage.

    Implementation for 590 speakers.

    Work of Lukic and Vogt, adapted by Heusser
"""
from lasagne import layers
from lasagne import nonlinearities

from .spectrogram_cnn import SpectrogramCnn


class SpectrogramCnn590(SpectrogramCnn):
    def create_paper(self, shape):
        paper = [
            # input layer
            (layers.InputLayer, {'shape': (None, shape, self.config.getint('luvo', 'spectrogram_height'),
                                           self.config.getint('luvo', 'seg_size'))}),

            # convolution layers 1
            (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (4, 4)}),
            (layers.MaxPool2DLayer, {'pool_size': (4, 4), 'stride': (2, 2)}),

            # convolution layers 2
            (layers.Conv2DLayer, {'num_filters': 64, 'filter_size': (4, 4)}),
            (layers.MaxPool2DLayer, {'pool_size': (4, 4), 'stride': (2, 2)}),

            # dense layer
            (layers.DenseLayer, {'num_units': 5900}),
            (layers.DropoutLayer, {}),
            (layers.DenseLayer, {'num_units': 2950}),

            # output layer
            (layers.DenseLayer, {'num_units': 590, 'nonlinearity': nonlinearities.softmax})
        ]

        return paper
