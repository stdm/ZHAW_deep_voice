import numpy as np
import mxnet as mx
import random

from random import randint
from common.utils.pickler import load
from networks.lstm_arc_face import settings 

class SimpleIter(mx.io.DataIter):
    def __init__(self, data_names, label_names, data_gen, num_batches=10):
        n = next(data_gen)
        self._provide_data = [(data_names, np.array(n[0]).shape)]
        self._provide_label = [(label_names, np.array(n[1]).shape)]
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.cur_batch = 0

    def reset(self):
        self.cur_batch = 0

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            n = next(self.data_gen)
            return mx.io.DataBatch([mx.nd.array(n[0], mx.gpu(0))], [mx.nd.array(n[1], mx.gpu(0))])
        else:
            raise StopIteration

def extract(spectrogram, segment_size, frequency_elements=settings.FREQ_ELEMENTS):
    zeros = 0

    for x in spectrogram[0]:
        if x == 0.0:
            zeros += 1
        else:
            zeros = 0

    while spectrogram.shape[1] - zeros < segment_size:
        zeros -= 1

    return spectrogram[0:frequency_elements, 0:spectrogram.shape[1] - zeros]

def _data_splitter(x, y, val_perc):
    zipped = list(zip(x, y))
    random.shuffle(zipped)

    idx = int(val_perc * len(zipped))
    train, val = zipped[idx:], zipped[:idx]

    x_t, y_t = zip(*train)
    x_v, y_v = zip(*val)
    return x_t, y_t, x_v, y_v

def batch_generator_lstm(X, y, batch_size=100, segment_size=15):
    X = np.array(list(X))
    y = np.array(list(y))
    segments = X.shape[0]
    bs = batch_size
    speakers = np.amax(y) + 1
    while 1:
        for i in range((segments + bs - 1) // bs):
            Xb = []
            yb = []
            for j in range(0, bs):
                speaker_idx = randint(0, len(X) - 1)
                if y is not None:
                    yb.append(y[speaker_idx])
                spect = extract(X[speaker_idx, 0], segment_size)
                seg_idx = randint(0, spect.shape[1] - segment_size)
                Xb.append(np.transpose(spect[:, seg_idx:seg_idx + segment_size]))
            yield Xb, yb
            
def get_iter(bg, batches_per_epoch):
    x, y = [], []
    bs = 0
    i = 0
    for xx, yy in bg:
        if i == batches_per_epoch:
            bs = len(xx)
            break
        x.extend(xx)
        y.extend(yy)
        i += 1
    return mx.io.NDArrayIter(data=np.array(x), label=np.array(y), batch_size=bs)

def load_data(data_path, batch_size, num_batches, val_perc=0.2):
    x, y, speaker_names = load(data_path)
    
    num_speakers = np.amax(y) + 1
    x_t, y_t, x_v, y_v = _data_splitter(x, y, val_perc)
    
    
    bg_t = batch_generator_lstm(x_t, y_t, batch_size)
    bg_v = batch_generator_lstm(x_v, y_v, batch_size)
    
    iter_t = SimpleIter(['data'], ['softmax_label'], bg_t, num_batches)
    iter_v = SimpleIter(['data'], ['softmax_label'], bg_v, num_batches)
    
    return iter_t, iter_v, num_speakers
