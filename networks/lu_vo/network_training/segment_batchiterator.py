from random import randint

import numpy as np
from nolearn.lasagne import BatchIterator

from common.spectogram.spectrogram_extractor import extract_spectrogram


class SegmentBatchIterator(BatchIterator):
    def __init__(self, batch_size, config):
        super().__init__(batch_size)
        self.config = config

    def __iter__(self):
        bs = self.batch_size
        seg_size = self.config.getint('luvo', 'seg_size')
        spectogram_height = self.config.getint('luvo', 'spectogram_height')
        # build as much batches as fit into the training set
        for i in range((self.n_samples + bs - 1) // bs):
            Xb = np.zeros((bs, 1, spectogram_height, seg_size), dtype=np.float32)
            yb = np.zeros(bs, dtype=np.int32)
            # here one batch is generated
            for j in range(0, bs):
                speaker_idx = randint(0, len(self.X) - 1)
                if self.y is not None:
                    yb[j] = self.y[speaker_idx]
                spectrogramm = extract_spectrogram(self.X[speaker_idx, 0], seg_size, spectogram_height)
                seg_idx = randint(0, spectrogramm.shape[1] - seg_size)
                Xb[j, 0] = spectrogramm[:, seg_idx:seg_idx + seg_size]
            yield Xb, yb
