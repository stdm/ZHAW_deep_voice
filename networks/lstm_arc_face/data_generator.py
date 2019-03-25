import numpy as np
import mxnet as mx
import random

from common.utils.pickler import load

def _data_splitter(x, y, val_perc):
    zipped = zip(x, y)
    random.shuffle(zipped)

    idx = int(val_perc * len(zipped))
    train, val = zipped[idx:], zipped[:idx]

    x_t, y_t = zip(*train)
    x_v, y_v = zip(*val)
    return x_t, y_t, x_v, y_v

def load_data(data_path, batch_size, val_perc=0.2):
    x, y, speaker_names = load(data_path)

    x_t, y_t, x_v, y_v = _data_splitter(x, y, val_perc)
    num_speakers = np.amax(y) + 1

    train_iter = mx.io.NDArrayIter(data=np.squeeze(x_t), label=np.array(y_t), batch_size=batch_size)
    val_iter = mx.io.NDArrayIter(data=np.squeeze(x_v), label=np.array(y_v), batch_size=batch_size)

    return train_iter, val_iter, num_speakers
