import pickle

import numpy as np


import common.spectrogram.speaker_train_splitter as sts
from .core import plot_saver as ps

np.random.seed(1337)  # for reproducibility

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional
from networks.pairwise_lstm.core import pairwise_kl_divergence as kld

from common.utils.paths import *

'''This Class Trains a Bidirectional LSTM with 2 Layers, and 2 Denselayer and a Dropout Layers
    Parameters:
    name: Network/experiment name to save artifacts
    training_data: name of the Training data file, expected to be train_xxxx.pickle
    n_hidden1: Units of the first LSTM Layer
    n_hidden2: Units of the second LSTM Layer
    n_classes: Amount of output classes (Speakers in Trainingset)
    n_10_batches: Number of Minibatches to train the Network (1 = 10 Minibatches)
    segment_size: Segment size that is used as input 100 equals 1 second with current Spectrogram extraction
    frequency: size of the frequency Dimension of the Input Spectrogram

    Work of Gerber and Glinski.
'''


class bilstm_2layer_dropout(object):
    def __init__(self, name, config, data_generator):
        self.network_name = name
        self.training_data = config.get('train', 'pickle')
        self.n_hidden1 = config.getint('pairwise_lstm', 'n_hidden1')
        self.n_hidden2 = config.getint('pairwise_lstm', 'n_hidden2')
        self.n_classes = config.getint('pairwise_lstm', 'n_classes')
        self.n_10_batches = config.getint('pairwise_lstm', 'n_10_batches')
        self.adam_lr = config.getfloat('pairwise_lstm', 'adam_lr')
        self.adam_beta_1 = config.getfloat('pairwise_lstm', 'adam_beta_1')
        self.adam_beta_2 = config.getfloat('pairwise_lstm', 'adam_beta_2')
        self.adam_epsilon = config.getfloat('pairwise_lstm', 'adam_epsilon')
        self.adam_decay = config.getfloat('pairwise_lstm', 'adam_decay')
        self.segment_size = config.getint('pairwise_lstm', 'seg_size')
        self.frequency = config.getint('pairwise_lstm', 'spectrogram_height')
        self.input = (self.segment_size, self.frequency)
        self.dg = data_generator
        print(self.network_name)
        self.run_network()

    def create_net(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(self.n_hidden1, return_sequences=True), input_shape=self.input))
        model.add(Dropout(0.50))
        model.add(Bidirectional(LSTM(self.n_hidden2)))
        model.add(Dense(self.n_classes * 10))
        model.add(Dropout(0.25))
        model.add(Dense(self.n_classes * 5))
        model.add(Dense(self.n_classes))
        model.add(Activation('softmax'))
        adam = keras.optimizers.Adam(self.adam_lr, self.adam_beta_1, self.adam_beta_2,
                                     self.adam_epsilon, self.adam_decay)
        model.compile(loss=kld.pairwise_kl_divergence,
                      optimizer=adam,
                      metrics=['accuracy'])
        return model

    def create_train_data(self):
        with open(get_speaker_pickle(self.training_data), 'rb') as f:
            (X, y, speaker_names) = pickle.load(f)

        splitter = sts.SpeakerTrainSplit(0.2)
        X_t, X_v, y_t, y_v = splitter(X, y)

        return X_t, y_t, X_v, y_v

    def create_callbacks(self):
        csv_logger = keras.callbacks.CSVLogger(get_experiment_logs(self.network_name + '.csv'))
        net_saver = keras.callbacks.ModelCheckpoint(
            get_experiment_nets(self.network_name + "_best.h5"),
            monitor='val_loss', verbose=1, save_best_only=True)
        net_checkpoint = keras.callbacks.ModelCheckpoint(
            get_experiment_nets(self.network_name + "_{epoch:05d}.h5"), period=self.n_10_batches / 10)
        return [csv_logger, net_saver, net_checkpoint]

    def run_network(self):
        model = self.create_net()
        calls = self.create_callbacks()

        X_t, y_t, X_v, y_v = self.create_train_data()
        train_gen = self.dg.batch_generator_divergence_optimised(X_t, y_t, 100, sentences=8)
        val_gen = self.dg.batch_generator_divergence_optimised(X_v, y_v, 100, sentences=2)
        # batches_t = ((X_t.shape[0] + 128 - 1) // 128)
        # batches_v = ((X_v.shape[0] + 128 - 1) // 128)

        history = model.fit_generator(train_gen, steps_per_epoch=10, epochs=self.n_10_batches,
                                      verbose=2, callbacks=calls, validation_data=val_gen,
                                      validation_steps=2, class_weight=None, max_q_size=10,
                                      nb_worker=1, pickle_safe=False)

        ps.save_accuracy_plot(history, self.network_name)
        ps.save_loss_plot(history, self.network_name)
        print("saving model")
        model.save(get_experiment_nets(self.network_name + ".h5"))
