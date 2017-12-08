"""
    This file provides a BatchIterator that supplies the minibatches to the spectrogram_cnn's.

    Work of Lukic and Vogt.
"""

from random import randint

import numpy as np
from nolearn.lasagne import BatchIterator

from networks.pairwise_kldiv.core import settings


class SegmentBatchIterator(BatchIterator):
    def __init__(self, batch_size):
        super(SegmentBatchIterator, self).__init__(batch_size)

    def __iter__(self):
        bs = self.batch_size
        # build as much batches as fit into the training set
        for i in range((self.n_samples + bs - 1) // bs):
            Xb = np.zeros((bs, 1, settings.FREQ_ELEMENTS, settings.ONE_SEC), dtype=np.float32)
            yb = np.zeros(bs, dtype=np.int32)
            # here one batch is generated
            for j in range(0, bs):
                speaker_idx = randint(0, len(self.X) - 1)
                if self.y is not None:
                    yb[j] = self.y[speaker_idx]
                spect = self.extract_spectrogram(self.X[speaker_idx, 0])
                seg_idx = randint(0, spect.shape[1] - settings.ONE_SEC)
                Xb[j, 0] = spect[:, seg_idx:seg_idx + settings.ONE_SEC]
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb

    @staticmethod
    def extract_spectrogram(spectrogram):
        zeros = 0
        for x in spectrogram[0]:
            if x == 0.0:
                zeros += 1
            else:
                zeros = 0
        while spectrogram.shape[1] - zeros < settings.ONE_SEC:
            zeros -= 1
        spect = spectrogram[0:settings.FREQ_ELEMENTS, 0:spectrogram.shape[1] - zeros]
        return spect


class DoubleSegmentBatchIterator(BatchIterator):
    def __init__(self, batch_size, minibatches_per_epoch):
        super(DoubleSegmentBatchIterator, self).__init__(batch_size)
        self.minibatches_per_epoch = minibatches_per_epoch

    def __iter__(self):
        bs = self.batch_size
        for i in range(self.minibatches_per_epoch):
            Xb = np.zeros((bs, 1, settings.FREQ_ELEMENTS * 2, settings.ONE_SEC), dtype=np.float32)
            yb = np.zeros(bs, dtype=np.int32)
            # here one batch is generated
            for j in range(0, bs):
                sentence1_idx = randint(0, len(self.X) - 1)
                if j % 2 == 0:
                    speaker_indexes = np.where(self.y == self.y[sentence1_idx])[0]
                    sentence2_idx = randint(speaker_indexes[0], speaker_indexes[len(speaker_indexes) - 1])
                else:
                    sentence2_idx = randint(0, len(self.X) - 1)
                if self.y is not None:
                    yb[j] = 1 if self.y[sentence1_idx] == self.y[sentence2_idx] else 0
                spect1 = self.extract_spectrogram(self.X[sentence1_idx, 0])
                seg_idx1 = randint(0, spect1.shape[1] - settings.ONE_SEC)
                if j % 2 == 0:
                    spect2 = spect1
                else:
                    spect2 = self.extract_spectrogram(self.X[sentence2_idx, 0])
                seg_idx2 = randint(0, spect2.shape[1] - settings.ONE_SEC)
                Xb[j, 0] = np.concatenate(
                    (spect1[:, seg_idx1:seg_idx1 + settings.ONE_SEC], spect2[:, seg_idx2:seg_idx2 + settings.ONE_SEC]),
                    axis=0)
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb

    @staticmethod
    def extract_spectrogram(spectrogram):
        zeros = 0
        for x in spectrogram[0]:
            if x == 0.0:
                zeros += 1
            else:
                zeros = 0
        while spectrogram.shape[1] - zeros < settings.ONE_SEC:
            zeros -= 1
        spect = spectrogram[0:settings.FREQ_ELEMENTS, 0:spectrogram.shape[1] - zeros]
        return spect
