from random import randint

import numpy as np
from nolearn.lasagne import BatchIterator

from common.spectogram.spectrogram_extractor import extract_spectrogram
from .. import settings


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
                spectrogramm = extract_spectrogram(self.X[speaker_idx, 0], settings.ONE_SEC, settings.FREQ_ELEMENTS)
                seg_idx = randint(0, spectrogramm.shape[1] - settings.ONE_SEC)
                Xb[j, 0] = spectrogramm[:, seg_idx:seg_idx + settings.ONE_SEC]
            yield Xb, yb
