import mxnet as mx
import numpy as np
from random import randint

class CustomIterator(mx.io.DataIter):
    def __init__(self, X, Y, batch_size):
        self.speakers = np.amax(Y) + 1
        self._provide_data = list(zip(['data'], tuple([batch_size] + list(X.shape[1:]))))
        self._provide_label = list(zip(['data'], tuple([batch_size] + list(Y.shape[1:]) + [self.speakers])))
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.leftover = None
        self.cur_step = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_step = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        next_step = self.cur_step + self.batch_size
        if next_step >= len(self.Y):
            x = []
            y = []
            for i in range(next_step - len(self.Y)):
                idx = self.cur_step + i
                x.append(self.X[idx])
                ym = np.zeros(self.speakers)
                ym[self.Y[idx]] = 1
                y.append(ym)
            self.leftover = [x, y]
            raise StopIteration
        else:
            x = []
            y = []
            if self.leftover is not None:
                x.extend(self.leftover[0])
                y.extend(self.leftover[1])
                self.leftover = None
            while len(y) < self.batch_size:
                x.append(self.X[self.cur_step])
                ym = np.zeros(self.speakers)
                ym[self.Y[self.cur_step]] = 1
                y.append(ym)
                self.cur_step += 1
            return mx.io.DataBatch([x], [y], self.batch_size)
