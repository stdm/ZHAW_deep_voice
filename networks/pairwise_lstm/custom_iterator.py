import mxnet as mx
import numpy as np

class CustomIterator(mx.io.NDArrayIter):
    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator."""
        desc = []
        for k, v in self.label:
            speakers = np.amax(v) + 1
            speakers = int(speakers.asnumpy()[0])
            desc.append(mx.io.DataDesc(k, tuple([self.batch_size] + list(v.shape[1:]) + [speakers]), v.dtype))
        return desc

    def getlabel(self):
        """Get label."""
        print(self.label[0])
        input('test')
        batch = self._batchify(self.label)
        while 1:
            for i in range((segments + bs - 1) // bs):
                Xb = np.zeros((bs, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
                yb = np.zeros(bs, dtype=np.int32)
                # here one batch is generated
                for j in range(0, bs):
                    speaker_idx = randint(0, len(X) - 1)
                    if y is not None:
                        yb[j] = y[speaker_idx]
                    spect = extract(X[speaker_idx, 0], segment_size)
                    seg_idx = randint(0, spect.shape[1] - segment_size)
                    Xb[j, 0] = spect[:, seg_idx:seg_idx + segment_size]
                yield Xb, create_pairs(yb)
        print(batch.shape)
        input('test')
        return batch
