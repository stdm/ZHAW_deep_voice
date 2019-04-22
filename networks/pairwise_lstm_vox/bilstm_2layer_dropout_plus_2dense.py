import pickle
import numpy as np

np.random.seed(1337)  # for reproducibility

import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers.wrappers import Bidirectional
from keras.wrappers.scikit_learn import KerasClassifier
from modAL.models import ActiveLearner

from .core import data_gen as dg
from .core import pairwise_kl_divergence as kld
from .core import plot_saver as ps

import common.spectogram.speaker_train_splitter as sts
from common.utils.paths import *

'''This Class Trains a Bidirectional LSTM with 2 Layers, and 2 Denselayer and a Dropout Layers
    Parameters:
    name: Network/experiment name to save artifacts
    training_data: name of the Training data file, expected to be train_xxxx.pickle
    n_hidden1: Units of the first LSTM Layer
    n_hidden2: Units of the second LSTM Layer
    n_classes: Amount of output classes (Speakers in Trainingset)
    epochs: Number of Epochs to train the Network per ActiveLearningRound
    segment_size: Segment size that is used as input 100 equals 1 second with current Spectrogram extraction
    frequency: size of the frequency Dimension of the Input Spectrogram

    Work of Gerber and Glinski.
'''


class bilstm_2layer_dropout(object):
    def __init__(self, name, training_data, n_hidden1, n_hidden2, n_classes, epochs, activeLearnerRounds,
                 segment_size, frequency=128):
        self.network_name = name
        self.training_data = training_data
        self.test_data = 'test' + training_data[5:]
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_classes = n_classes
        self.epochs = epochs
        self.activeLearnerRounds = activeLearnerRounds
        self.segment_size = segment_size
        self.input = (segment_size, frequency)
        print(self.network_name)
        self.run_network()

    def create_net(self):
        # tensorflow_gpus_available = len(backend.tensorflow_backend._get_available_gpus()) > 0
        
        model = Sequential()

        # LSTM
        #model.add(Bidirectional(CuDNNLSTM(self.n_hidden1, return_sequences=True), input_shape=self.input))
        model.add(Bidirectional(LSTM(self.n_hidden1, return_sequences=True), input_shape=self.input))

        model.add(Dropout(0.50))

        # LSTM
        #model.add(Bidirectional(CuDNNLSTM(self.n_hidden2)))
        model.add(Bidirectional(LSTM(self.n_hidden2)))

        model.add(Dense(self.n_classes * 10))
        model.add(Dropout(0.25))
        model.add(Dense(self.n_classes * 5))
        model.add(Dense(self.n_classes))
        model.add(Activation('softmax'))
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(loss=kld.pairwise_kl_divergence, optimizer=adam, metrics=['accuracy'])
        print(model)
        return model

    # This method splits the given training data into training and validation
    # sets for the training step
    #
    def create_train_data(self, activeLearningRound):
        training_data_round = self.training_data + '_al_' + str(activeLearningRound)

        print('create_train_data', self.training_data, 'AL round', activeLearningRound)
        print('create_train_data', get_speaker_pickle(training_data_round))

        with open(get_speaker_pickle(training_data_round), 'rb') as f:
            (X, y, speaker_names) = pickle.load(f)

        splitter = sts.SpeakerTrainSplit(0.2)
        X_t, X_v, y_t, y_v, _speaker_t, _speaker_v = splitter(X, y, speaker_names)
        return X_t, y_t, X_v, y_v

    def create_callbacks(self):
        csv_logger = keras.callbacks.CSVLogger(
            get_experiment_logs(self.network_name + '.csv'))
        net_saver = keras.callbacks.ModelCheckpoint(
            get_experiment_nets(self.network_name + "_best.h5"),
            monitor='val_loss', verbose=1, save_best_only=True)
        net_checkpoint = keras.callbacks.ModelCheckpoint(
            get_experiment_nets(self.network_name + "_{epoch:05d}.h5"), period=100)

        return [csv_logger, net_saver, net_checkpoint]

    def run_network(self):
        # base keras network
        model = self.create_net()
        calls = self.create_callbacks()

        # wrappers for AL
        classifier = KerasClassifier(model)
        learner = ActiveLearner(estimator=classifier, verbose=1)

        # initial train set
        X_t, y_t, X_v, y_v = self.create_train_data(0)

        for i in range(self.activeLearnerRounds): # ActiveLearning Rounds
            if i != 0:
                # query active learner for uncertainty based on pool and append to numpy
                X_t_pool, y_t_pool, X_v_pool, y_v_pool = self.create_train_data(i)
                query_idx, _ = learner.query(X_t_pool, n_instances=100, verbose=0)
                
                np.append(X_t, X_t_pool[query_idx])
                np.append(y_t, y_t_pool[query_idx])
                np.append(X_v, X_v_pool[query_idx])
                np.append(y_v, y_v_pool[query_idx])

            # TODO lehmacl1@2019-03-05: Müssen hier nicht 2er Potenzen als Batchsize (100) mitgegeben werden?
            train_gen = dg.batch_generator_lstm(X_t, y_t, 100, segment_size=self.segment_size)
            val_gen = dg.batch_generator_lstm(X_v, y_v, 100, segment_size=self.segment_size)
            # batches_t = ((X_t.shape[0] + 128 - 1) // 128)
            # batches_v = ((X_v.shape[0] + 128 - 1) // 128)

            # NOTE: lehmacl1@2019-04-14: bilstm_2layer_dropout_plus_2dense.py:113: UserWarning:
            # The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1
            # argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the
            # generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size.
            #
            # Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps`
            # arguments have changed. Update your method calls accordingly.

            # NOTE: lehmacl1@2019-04-14: bilstm_2layer_dropout_plus_2dense.py:113: UserWarning:
            # Update your `fit_generator` call to the Keras 2 API:
            # `fit_generator(<generator..., callbacks=[<keras.ca..., use_multiprocessing=False,
            # class_weight=None, epochs=1000, workers=1, steps_per_epoch=10, validation_steps=2,
            # verbose=2, validation_data=<generator..., max_queue_size=10)`

            history = model.fit_generator(
                train_gen, 
                steps_per_epoch=10, 
                epochs=self.epochs,
                verbose=2, 
                callbacks=calls, 
                validation_data=val_gen,
                validation_steps=2, 
                class_weight=None, 
                max_q_size=10,
                nb_worker=1, 
                pickle_safe=False
            )

            ps.save_accuracy_plot(history, self.network_name)
            ps.save_loss_plot(history, self.network_name)
            print("saving model")
            model.save(get_experiment_nets(self.network_name + ".h5"))
            # print "evaluating model"
            # da.calculate_test_acccuracies(self.network_name, self.test_data, True, True, True, segment_size=self.segment_size)
