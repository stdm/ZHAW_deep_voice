import mxnet as mx
import numpy as np

class CustomIterator(mx.io.NDArrayIter):
    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator."""
        desc = []
        for k, v in self.label:
            speakers = np.amax(v) + 1
            desc.append(mx.io.DataDesc(k, tuple([self.batch_size] + list(v.shape[1:]) + [speakers]), v.dtype))
        print(desc)
        input('test')
        return desc

    def getlabel(self):
        """Get label."""
        batch = self._batchify(self.label)
        print(batch.shape)
        input('test')
        return batch
