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

    def _getdata(self, data_source, start=None, end=None):
        """Load data from underlying arrays."""
        print(data_source)
        input('test')
        assert start is not None or end is not None, 'should at least specify start or end'
        start = start if start is not None else 0
        if end is None:
            end = data_source[0][1].shape[0] if data_source else 0
        s = slice(start, end)
        return [
            x[1][s]
            if isinstance(x[1], (np.ndarray, NDArray)) else
            # h5py (only supports indices in increasing order)
            array(x[1][sorted(self.idx[s])][[
                list(self.idx[s]).index(i)
                for i in sorted(self.idx[s])
            ]]) for x in data_source
        ]

    def _batchify(self, data_source):
        """Load data from underlying arrays, internal use only."""
        assert self.cursor < self.num_data, 'DataIter needs reset.'
        # first batch of next epoch with 'roll_over'
        input('1')
        if self.last_batch_handle == 'roll_over' and \
            -self.batch_size < self.cursor < 0:
            input('2')
            assert self._cache_data is not None or self._cache_label is not None, \
                'next epoch should have cached data'
            cache_data = self._cache_data if self._cache_data is not None else self._cache_label
            second_data = self._getdata(
                data_source, end=self.cursor + self.batch_size)
            if self._cache_data is not None:
                self._cache_data = None
            else:
                self._cache_label = None
            return self._concat(cache_data, second_data)
        # last batch with 'pad'
        elif self.last_batch_handle == 'pad' and \
            self.cursor + self.batch_size > self.num_data:
            input('3')
            pad = self.batch_size - self.num_data + self.cursor
            first_data = self._getdata(data_source, start=self.cursor)
            second_data = self._getdata(data_source, end=pad)
            return self._concat(first_data, second_data)
        # normal case
        else:
            input('4')
            if self.cursor + self.batch_size < self.num_data:
                end_idx = self.cursor + self.batch_size
            # get incomplete last batch
            else:
                end_idx = self.num_data
            return self._getdata(data_source, self.cursor, end_idx)

    def getlabel(self):
        """Get label."""
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
        return batch
