"""
    The batch creation file to create generators that yield the batches.

    Work of Gerber and Glinski.
"""
from random import randint

import numpy as np

from common.spectogram.spectrogram_extractor import extract_spectrogram
from . import settings


def transform(Xb, yb):
    return Xb, yb


# Shuffles both array along the first Dimensions, in a way that they end up at the same position.
def shuffle_data(Xb, yb):
    rng_state = np.random.get_state()
    np.random.shuffle(Xb)
    np.random.set_state(rng_state)
    np.random.shuffle(yb)


# Extracts the Spectorgram and discards all padded Data
def extract(spectrogram, segment_size):
    return extract_spectrogram(spectrogram, segment_size, settings.FREQ_ELEMENTS)


# generates the data for testing the network, with the specified segment_size (timewindow)
def generate_test_data(X, y, segment_size=15):
    segments = X.shape[0] * 3 * (800 // segment_size)
    X_test = np.zeros((segments, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
    y_test = []

    pos = 0
    for i in range(len(X)):
        spect = extract(X[i, 0], segment_size)

        for j in range(int(spect.shape[1] / segment_size)):
            y_test.append(y[i])
            seg_idx = j * segment_size
            X_test[pos, 0] = spect[:, seg_idx:seg_idx + segment_size]
            pos += 1

    return X_test[0:len(y_test)], np.asarray(y_test, dtype=np.int32)


# Batch generator for CNNs
def batch_generator(X, y, batch_size=100, segment_size=100):
    segments = X.shape[0]
    bs = batch_size
    speakers = np.amax(y) + 1
    # build as much batches as fit into the training set
    while 1:
        for i in range((segments + bs - 1) // bs):
            Xb = np.zeros((bs, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
            yb = np.zeros(bs, dtype=np.int32)
            # here one batch is generated
            for j in range(0, bs):
                speaker_idx = randint(0, len(X) - 1)
                if y is not None:
                    yb[j] = y[speaker_idx]
                spect = extract(X[speaker_idx, 0], segment_size)
                seg_idx = randint(0, spect.shape[1] - segment_size)
                Xb[j, 0] = spect[:, seg_idx:seg_idx + segment_size]
            yield Xb, transformy(yb, bs, speakers)


'''creates the a batch for CNN networks, with Pariwise Labels, 
for use with core.pairwise_kl_divergence_full_labels'''


def batch_generator_v2(X, y, batch_size=100, segment_size=100):
    segments = X.shape[0]
    bs = batch_size
    speakers = np.amax(y) + 1
    # build as much batches as fit into the training set
    while 1:
        for i in range((segments + bs - 1) // bs):
            Xb = np.zeros((bs, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
            yb = np.zeros(bs, dtype=np.int32)
            # here one batch is generated
            for j in range(0, bs):
                speaker_idx = randint(0, len(X) - 1)
                if y is not None:
                    yb[j] = y[speaker_idx]
                spect = extract(X[speaker_idx, 0], segment_size)
                seg_idx = randint(0, spect.shape[1] - segment_size)
                Xb[j, 0] = spect[:, seg_idx:seg_idx + segment_size]
            yield Xb, create_pairs(yb)


# Batch generator von LSTMS
def batch_generator_lstm(X, y, batch_size=100, segment_size=15):
    segments = X.shape[0]
    bs = batch_size
    speakers = np.amax(y) + 1
    # build as much batches as fit into the training set
    while 1:
        for i in range((segments + bs - 1) // bs):
            Xb = np.zeros((bs, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
            yb = np.zeros(bs, dtype=np.int32)
            # here one batch is generated
            for j in range(0, bs):
                speaker_idx = randint(0, len(X) - 1)
                if y is not None:
                    yb[j] = y[speaker_idx]
                spect = extract(X[speaker_idx, 0], segment_size)
                seg_idx = randint(0, spect.shape[1] - segment_size)
                Xb[j, 0] = spect[:, seg_idx:seg_idx + segment_size]
            yield Xb.reshape(bs, segment_size, settings.FREQ_ELEMENTS), transformy(yb, bs, speakers)


'''creates the a batch for LSTM networks, with Pairwise Laabels, 
for use with core.pairwise_kl_divergence_full_labels'''


def batch_generator_lstm_v2(X, y, batch_size=100, segment_size=15):
    segments = X.shape[0]
    bs = batch_size
    speakers = np.amax(y) + 1
    # build as much batches as fit into the training set
    while 1:
        for i in range((segments + bs - 1) // bs):
            Xb = np.zeros((bs, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
            yb = np.zeros(bs, dtype=np.int32)
            # here one batch is generated
            for j in range(0, bs):
                speaker_idx = randint(0, len(X) - 1)
                if y is not None:
                    yb[j] = y[speaker_idx]
                spect = extract(X[speaker_idx, 0], segment_size)
                seg_idx = randint(0, spect.shape[1] - segment_size)
                Xb[j, 0] = spect[:, seg_idx:seg_idx + segment_size]
            yield Xb.reshape(bs, segment_size, settings.FREQ_ELEMENTS), create_pairs(yb)


def transformy(y, batch_size, nb_classes):
    yn = np.zeros((batch_size, nb_classes))
    k = 0
    for v in y:
        # print v
        yn[k][v] = 1
        k += 1
    return yn


def create_pairs(l):
    pair_list = []
    for i in range(len(l)):
        j = i + 1
        for j in range(i + 1, len(l)):
            print(j)
            if (l[i] == l[j]):
                pair_list.append((int(i), int(j), 1))
            else:
                pair_list.append((int(i), int(j), 0))
    return np.vstack(pair_list)


# Extracts the provided amount of samples from a Sentence
# The Samples are randomly chosen for each sentence.
# Example if there are 8 Sentences for 10 speakers each and the amount of samples is 3
# The function will return a numpy array Xb with shape 240, 1, 128, 100 and a Numpy array yb with shape 240
# this is for use in Keras model.fit function (does not yield good Training results)
def createData(X, y, samples, segment_size=15):
    segments = X.shape[0]
    idx = 0
    Xb = np.zeros((segments * samples, 1, settings.FREQ_ELEMENTS, segment_size), dtype=np.float32)
    yb = np.zeros(segments * samples, dtype=np.int32)
    for i in range(segments):
        # here one batch is generated
        for j in range(0, samples):
            speaker_idx = y[i]
            yb[idx] = speaker_idx
            spect = extract(X[i, 0], segment_size)
            seg_idx = randint(0, spect.shape[1] - segment_size)
            Xb[idx, 0] = spect[:, seg_idx:seg_idx + segment_size]
            idx += 1
    shuffle_data(Xb, yb)
    return Xb, yb
