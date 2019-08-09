"""
    The batch creation file to create generators that yield the batches.

    Work of Gerber and Glinski.
"""
from random import randint, sample

import numpy as np

from common.spectrogram.spectrogram_extractor import extract_spectrogram


class DataGenerator:
    def __init__(self, segment_size=100, spectrogram_height=128):
        self.segment_size = segment_size
        self.spectrogram_height = spectrogram_height

    # generates the data for testing the network, with the specified segment_size (timewindow)
    def generate_test_data(self, X, y):
        segments = X.shape[0] * 3 * (800 // self.segment_size)
        X_test = np.zeros((segments, 1, self.spectrogram_height, self.segment_size), dtype=np.float32)
        y_test = []

        pos = 0
        for i in range(len(X)):
            spect = self._extract(X[i, 0])

            for j in range(int(spect.shape[1] / self.segment_size)):
                y_test.append(y[i])
                seg_idx = j * self.segment_size
                X_test[pos, 0] = spect[:, seg_idx:seg_idx + self.segment_size]
                pos += 1

        return X_test[0:len(y_test)], np.asarray(y_test, dtype=np.int32)

    # Batch generator for CNNs
    def batch_generator_cnn(self, X, y, batch_size=100):
        segments = X.shape[0]
        bs = batch_size
        speakers = np.amax(y) + 1
        # build as much batches as fit into the training set
        while 1:
            for i in range((segments + bs - 1) // bs):
                Xb = np.zeros((bs, 1, self.spectrogram_height, self.segment_size), dtype=np.float32)
                yb = np.zeros(bs, dtype=np.int32)
                # here one batch is generated
                for j in range(0, bs):
                    speaker_idx = randint(0, len(X) - 1)
                    if y is not None:
                        yb[j] = y[speaker_idx]
                    spect = self._extract(X[speaker_idx, 0])
                    seg_idx = randint(0, spect.shape[1] - self.segment_size)
                    Xb[j, 0] = spect[:, seg_idx:seg_idx + self.segment_size]
                yield Xb.reshape(bs,X.shape[1], self.segment_size, self.spectrogram_height), self._transformy(yb, bs, speakers)

    # Batch generator von LSTMS
    def batch_generator_lstm(self, X, y, batch_size=100):
        segments = X.shape[0]
        bs = batch_size
        speakers = np.amax(y) + 1
        # build as much batches as fit into the training set
        while 1:
            for i in range((segments + bs - 1) // bs):
                Xb = np.zeros((bs, 1, self.spectrogram_height, self.segment_size), dtype=np.float32)
                yb = np.zeros(bs, dtype=np.int32)
                # here one batch is generated
                for j in range(0, bs):
                    speaker_idx = randint(0, len(X) - 1)
                    if y is not None:
                        yb[j] = y[speaker_idx]
                    spect = self._extract(X[speaker_idx, 0])
                    seg_idx = randint(0, spect.shape[1] - self.segment_size)
                    Xb[j, 0] = spect[:, seg_idx:seg_idx + self.segment_size]

                yield Xb.reshape(bs, self.segment_size, self.spectrogram_height), self._transformy(yb, bs, speakers)

    def batch_generator_divergence_optimised(self, X, y, batch_size=100, sentences=8):
        segments = X.shape[0]
        bs = batch_size
        speakers = np.amax(y) + 1
        # build as much batches as fit into the training set
        while 1:
            for i in range((segments + bs - 1) // bs):
                # prepare arrays
                Xb = np.zeros((bs, 1, self.spectrogram_height, self.segment_size), dtype=np.float32)
                yb = np.zeros(bs, dtype=np.int32)
                #choose max. 100 speakers from all speakers contained in X (no duplicates!)
                population = set(y)
                n_speakers = min(len(population), 100)
                samples = sample(population, n_speakers)
                # here one batch is generated
                for j in range(0, bs):
                    #choose random sentence of one speaker out of the 100 sampled above (duplicates MUST be allowed here!)
                    #calculate the index of the sentence in X and y to access the data
                    speaker_id = randint(0, len(samples) - 1)
                    speaker_idx = sentences*speaker_id - randint(0,sentences-1)
                    if y is not None:
                        yb[j] = y[speaker_idx]
                    spect = self._extract(X[speaker_idx, 0])
                    seg_idx = randint(0, spect.shape[1] - self.segment_size)
                    Xb[j, 0] = spect[:, seg_idx:seg_idx + self.segment_size]
                yield Xb.reshape(bs, self.segment_size, self.spectrogram_height), self._transformy(yb, bs, speakers)

    # Extracts the Spectorgram and discards all padded Data
    def _extract(self, spectrogram):
        return extract_spectrogram(spectrogram, self.segment_size, self.spectrogram_height)

    @staticmethod
    def _transformy(y, batch_size, nb_classes):
        yn = np.zeros((batch_size, int(nb_classes)))
        k = 0
        for v in y:
            # print v
            yn[k][v] = 1
            k += 1
        return yn
