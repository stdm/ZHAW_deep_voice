import mxnet as mx
import numpy as np

class CustomIterator(mx.io.NDArrayIter):
    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator."""
        desc = []
        for k, v in self.label:
            speakers = np.amax(v) + 1
            speakers = int(speakers.tolist()[0])
            print('k:')
            print(k)
            print()
            print('speakers:')
            print(speakers)
            print()
            print('shape:')
            print(tuple([self.batch_size] + [speakers]))
            print()
            input('test')
            desc.append(mx.io.DataDesc(k, tuple([self.batch_size] + list(v.shape[1:]) + [speakers]), v.dtype))
        return desc

    def getlabel(self):
        """Get label."""
        batch = self._batchify(self.label)
        print(batch.shape)
        input('test')
        return batch
