import mxnet as mx
import numpy as np

def prepare_labels(labels):
    speakers = np.amax(labels) + 1
    new_labels = []
    for l in labels:
        y = np.zeros(speakers)
        y[randint(0, speakers-1)] = 1
        new_labels.append(y)
    new_labels = np.array(new_labels)

class CustomIterator(mx.io.NDArrayIter):
    def getlabel(self):
        """Get label."""
        batch = self._batchify(self.label)
        print(batch)
        input('wut')
        return batch
