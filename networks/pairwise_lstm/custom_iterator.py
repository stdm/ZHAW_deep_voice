import mxnet as mx
import numpy as np

class CustomIterator(mx.io.NDArrayIter):
    def __init__(self, data, label=None, batch_size=1, shuffle=False, last_batch_handle='pad', data_name='data', label_name='softmax_label'):
        super(CustomIterator, self).__init__(data, label, batch_size, shuffle, last_batch_handle, data_name, label_name)
        self.speakers = 0
        for k, v in self.label:
            for n in v:
                if self.speakers < n:
                    self.speakers = n
        self.speakers += 1

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator."""
        desc = []
        for k, v in self.label:
            print('k:')
            print(k)
            print()
            print('speakers:')
            print(self.speakers)
            print()
            print('shape:')
            print(tuple([self.batch_size] + [self.speakers]))
            print()
            input('test')
            desc.append(mx.io.DataDesc(k, tuple([self.batch_size] + list(v.shape[1:]) + [self.speakers]), v.dtype))
        return desc

    def getlabel(self):
        """Get label."""
        batch = self._batchify(self.label)
        print(batch.shape)
        input('test')
        return batch
