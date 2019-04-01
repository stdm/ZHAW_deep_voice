import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import random

from random import randint
from common.utils.pickler import load
from common.utils.paths import *

class SimpleIter(mx.io.DataIter):
    def __init__(self, data_gen, settings):
        n = next(data_gen)
        self._provide_data = [('data', np.array(n[0]).shape)]
        self._provide_label = [('softmax_label', np.array(n[1]).shape)]
        self.num_batches = settings['BATCHES_PER_EPOCH']
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
            return mx.io.DataBatch([n[0]], [n[1]])
        else:
            raise StopIteration

def _extract(spectrogram, settings):
    zeros = 0
    for x in spectrogram[0]:
        if x == 0.0:
            zeros += 1
        else:
            zeros = 0
    while spectrogram.shape[1] - zeros < settings['SEGMENT_SIZE']:
        zeros -= 1
    return spectrogram[0:settings['FREQ_ELEMENTS'], 0:spectrogram.shape[1] - zeros]

def _data_splitter(x, y, settings):
    sample_dic = {}
    zipped = list(zip(x, y))
    for X, Y in zipped:
        if Y not in sample_dic:
            sample_dic[Y] = []
        sample_dic[Y].append(X)

    x_t, x_v, y_t, y_v = [], [], [], []
    for Y in sample_dic:
        l = sample_dic[Y]
        random.shuffle(l)
        idx = int(settings['VAL_PERC'] * len(l))
        t, v = l[idx:], l[:idx]
        x_t.extend(t)
        x_v.extend(v)
        y_t.extend([Y]*len(t))
        y_v.extend([Y]*len(v))

    zipped = list(zip(x_t, y_t))
    random.shuffle(zipped)
    x_t, y_t = zip(*zipped)

    zipped = list(zip(x_v, y_v))
    random.shuffle(zipped)
    x_v, y_v = zip(*zipped)
    return x_t, y_t, x_v, y_v

def _batch_generator_lstm(X, y, settings):
    X = nd.array(list(X), mx.gpu(0))
    y = np.array(list(y))
    speakers = np.amax(y) + 1
    y = nd.array(list(y), mx.gpu(0))
    segments = X.shape[0]
    bs = settings['BATCH_SIZE']
    Xb = nd.zeros(bs, mx.gpu(0))
    yb = nd.zeros(bs, mx.gpu(0))
    while 1:
        for i in range((segments + bs - 1) // bs):
            for j in range(0, bs):
                speaker_idx = randint(0, len(X) - 1)
                if y is not None:
                    yb[j] = y[speaker_idx]
                spect = _extract(X[speaker_idx, 0], settings)
                print(spect)
                seg_idx = randint(0, spect.shape[1] - settings['SEGMENT_SIZE'])
                Xb[j] = nd.transpose(spect[:, seg_idx:seg_idx + settings['SEGMENT_SIZE']])
            yield Xb, yb

def load_test_data(settings):
    x_train, speakers_train = _load_test_data(get_speaker_pickle(settings['VAL_DATA_NAME'] + "_train"), settings)
    x_test, speakers_test = _load_test_data(get_speaker_pickle(settings['VAL_DATA_NAME'] + "_test"), settings)
    return x_train, speakers_train, x_test, speakers_test

def _load_test_data(data_path, settings):
    X, y, s_list = load(data_path)
    segments = X.shape[0] * 3 * (800 // settings['SEGMENT_SIZE'])
    X_test = np.zeros((segments, 1, settings['FREQ_ELEMENTS'], settings['SEGMENT_SIZE']), dtype=np.float32)
    y_test = []

    pos = 0
    for i in range(len(X)):
        spect = _extract(X[i, 0], settings)

        for j in range(int(spect.shape[1] / settings['SEGMENT_SIZE'])):
            y_test.append(y[i])
            seg_idx = j * settings['SEGMENT_SIZE']
            X_test[pos, 0] = spect[:, seg_idx:seg_idx + settings['SEGMENT_SIZE']]
            pos += 1

    x = X_test[0:len(y_test)]
    return x.reshape(x.shape[0], x.shape[3], x.shape[2]), np.asarray(y_test, dtype=np.int32)

def load_train_data(settings):
    x, y, speaker_names = load(get_speaker_pickle(settings['TRAIN_DATA_NAME']))
    x_t, y_t, x_v, y_v = _data_splitter(x, y, settings)
    num_speakers = np.amax(y) + 1

    bg_t = _batch_generator_lstm(x_t, y_t, settings)
    bg_v = _batch_generator_lstm(x_v, y_v, settings)

    iter_t = SimpleIter(bg_t, settings)
    iter_v = SimpleIter(bg_v, settings)

    return iter_t, iter_v, num_speakers
