import mxnet as mx
import numpy as np

class CustomIterator(mx.io.NDArrayIter):
    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator."""
        desc = []
        speakers = np.amax(y) + 1

        for k, v in self.label:
            print(tuple([self.batch_size] + list(v.shape[1:]) + [speakers]))
            input('test')
            desc.append(mx.io.DataDesc(k, tuple([self.batch_size] + list(v.shape[1:])), v.dtype))
        return desc
