import mxnet as mx

class CustomIterator(mx.io.NDArrayIter):
    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator."""
        desc = []
        for k, v in self.label:
            print(tuple([self.batch_size] + list(v.shape[1:])))
            input('test')
            desc.append(mx.io.DataDesc(k, tuple([self.batch_size] + list(v.shape[1:])), v.dtype))
        return desc
