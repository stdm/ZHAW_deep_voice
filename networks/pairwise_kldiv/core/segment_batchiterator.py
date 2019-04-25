"""
    This file provides a BatchIterator that supplies the minibatches to the spectrogram_cnn's.

    Work of Lukic and Vogt.
"""

from random import randint

import numpy as np
from nolearn.lasagne import BatchIterator


class SegmentBatchIterator(BatchIterator):
    def __init__(self, batch_size, config):
        super().__init__(batch_size)
        self.config = config

    def __iter__(self):
        bs = self.batch_size
        seg_size = self.config.getint('pairwise_kldiv', 'seg_size')
        spectrogram_height = self.config.getint('pairwise_kldiv', 'spectrogram_height')
        # build as much batches as fit into the training set
        for i in range((self.n_samples + bs - 1) // bs):
            Xb = np.zeros((bs, 1, spectrogram_height, seg_size), dtype=np.float32)
            yb = np.zeros(bs, dtype=np.int32)
            # here one batch is generated
            for j in range(0, bs):
                speaker_idx = randint(0, len(self.X) - 1)
                if self.y is not None:
                    yb[j] = self.y[speaker_idx]
                spect = self.extract_spectrogram(self.X[speaker_idx, 0])
                seg_idx = randint(0, spect.shape[1] - seg_size)
                Xb[j, 0] = spect[:, seg_idx:seg_idx + seg_size]
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb

    def _extract_spectrogram(self, spectrogram):
        zeros = 0
        seg_size = self.config.getint('pairwise_kldiv', 'seg_size')
        spectrogram_height = self.config.getint('pairwise_kldiv', 'spectrogram_height')
        for x in spectrogram[0]:
            if x == 0.0:
                zeros += 1
            else:
                zeros = 0
        while spectrogram.shape[1] - zeros < seg_size:
            zeros -= 1
        spect = spectrogram[0:spectrogram_height, 0:spectrogram.shape[1] - zeros]
        return spect


class DoubleSegmentBatchIterator(BatchIterator):
    def __init__(self, batch_size, minibatches_per_epoch, config):
        super().__init__(batch_size)
        self.minibatches_per_epoch = minibatches_per_epoch
        self.config = config

    def __iter__(self):
        bs = self.batch_size
        seg_size = self.config.getint('pairwise_kldiv', 'seg_size')
        spectrogram_height = self.config.getint('pairwise_kldiv', 'spectrogram_height')

        for i in range(self.minibatches_per_epoch):
            Xb = np.zeros((bs, 1, spectrogram_height * 2, seg_size), dtype=np.float32)
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
                seg_idx1 = randint(0, spect1.shape[1] - seg_size)
                if j % 2 == 0:
                    spect2 = spect1
                else:
                    spect2 = self.extract_spectrogram(self.X[sentence2_idx, 0])
                seg_idx2 = randint(0, spect2.shape[1] - seg_size)
                Xb[j, 0] = np.concatenate(
                    (spect1[:, seg_idx1:seg_idx1 + seg_size], spect2[:, seg_idx2:seg_idx2 + seg_size]),
                    axis=0)
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb

    def _extract_spectrogram(self, spectrogram):
        zeros = 0
        seg_size = self.config.getint('pairwise_kldiv', 'seg_size')
        spectrogram_height = self.config.getint('pairwise_kldiv', 'spectrogram_height')

        for x in spectrogram[0]:
            if x == 0.0:
                zeros += 1
            else:
                zeros = 0
        while spectrogram.shape[1] - zeros < seg_size:
            zeros -= 1
        spect = spectrogram[0:spectrogram_height, 0:spectrogram.shape[1] - zeros]
        return spect
