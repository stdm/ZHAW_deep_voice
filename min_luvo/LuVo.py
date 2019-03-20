import pickle

from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet, TrainSplit
import numpy as np
from nolearn.lasagne import BatchIterator
from random import randint

ONE_SEC = 50 # spectrogram width
FREQ_ELEMENTS = 128 #spectrogram height

class SegmentBatchIterator(BatchIterator):
    def __init__(self, batch_size):
        super(SegmentBatchIterator, self).__init__(batch_size)

    def __iter__(self):
        bs = self.batch_size
        # build as much batches as fit into the training set
        for i in range((self.n_samples + bs - 1) // bs):
            Xb = np.zeros((bs, 1, FREQ_ELEMENTS, ONE_SEC), dtype=np.float32)
            yb = np.zeros(bs, dtype=np.int32)
            # here one batch is generated
            for j in range(0, bs):
                speaker_idx = randint(0, len(self.X) - 1)
                if self.y is not None:
                    yb[j] = self.y[speaker_idx]
                spectrogramm = self.extract_spectrogram(self.X[speaker_idx, 0], ONE_SEC, FREQ_ELEMENTS)
                seg_idx = randint(0, spectrogramm.shape[1] - ONE_SEC)
                Xb[j, 0] = spectrogramm[:, seg_idx:seg_idx + ONE_SEC]
            yield Xb, yb

    @staticmethod
    def extract_spectrogram(spectrogram, segment_size, frequency_elements):
        zeros = 0
        for x in spectrogram[0]:
            if x == 0.0:
                zeros += 1
            else:
                zeros = 0

        while spectrogram.shape[1] - zeros < segment_size:
            zeros -= 1

        return spectrogram[0:frequency_elements, 0:spectrogram.shape[1] - zeros]

# load training data
with open('speakers_590_clustering_without_raynolds_train.pickle', 'rb') as f:
    X, y, speaker_names = pickle.load(f)

paper = [
    # input layer
    (layers.InputLayer, {'shape': (None, X.shape[1], FREQ_ELEMENTS, ONE_SEC)}),

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


# Setup the neural network
net = NeuralNet(
    layers=paper,

    # learning rate parameters
    update_learning_rate=0.001,
    update_momentum=0.9,
    regression=False,

    max_epochs=1000,
    verbose=1,
)

# Set new batch iterator
net.batch_iterator_train = SegmentBatchIterator(batch_size=128)
net.batch_iterator_test = SegmentBatchIterator(batch_size=128)
net.train_split = TrainSplit(eval_size=0)

# Train the network
print("Fitting...")
net.fit(X, y);