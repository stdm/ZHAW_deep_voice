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
    activeLearnerRounds: Number of learning rounds to requery the pool for new data
    segment_size: Segment size that is used as input 100 equals 1 second with current Spectrogram extraction
    frequency: size of the frequency Dimension of the Input Spectrogram

    Work of Gerber and Glinski.
'''


class bilstm_2layer_dropout(object):
    def __init__(self, name, training_data, n_hidden1, n_hidden2, n_classes, 
                 epochs, activeLearnerRounds, activeLearnerPools,
                 segment_size, frequency=128):
        self.network_name = name
        self.training_data = training_data
        self.test_data = 'test' + training_data[5:]
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_classes = n_classes
        self.epochs = epochs
        self.activeLearnerRounds = activeLearnerRounds
        self.activeLearnerPools = activeLearnerPools
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
        al_pool = activeLearningRound % self.activeLearnerPools
        training_data_round = self.training_data + '_al_' + str(al_pool)
        print('create_train_data', self.training_data, 'pool', al_pool)
        speaker_pickle = get_speaker_pickle(training_data_round)
        print('create_train_data', speaker_pickle)

        with open(speaker_pickle, 'rb') as f:
            (X, y, speaker_names) = pickle.load(f)

        return (X, y, speaker_pickle)

    def split_train_val_data(self, X, y):
        splitter = sts.SpeakerTrainSplit(0.2)
        X_t, X_v, y_t, y_v = splitter(X, y)
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
        known = dict()

        # initial train set
        X_pool, y_pool, pool_ident = self.create_train_data(0)
        known[pool_ident] = np.array(range(len(X_pool))
        # split
        X_t, y_t, X_v, y_v = self.split_train_val_data(X_pool, y_pool)

        for i in range(self.activeLearnerRounds): # ActiveLearning Rounds
            if i != 0:
                # query for uncertainty based on pool and append to numpy
                self.active_learning_round(model, known, i, X_t, X_v, y_t, y_v)

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
                callbacks=calls, 
                validation_data=val_gen,
                validation_steps=2, 
                class_weight=None, 
                max_q_size=10,
                nb_worker=1, 
                pickle_safe=False,
                verbose=2
            )

            ps.save_accuracy_plot(history, self.network_name)
            ps.save_loss_plot(history, self.network_name)
            print("saving model")
            model.save(get_experiment_nets(self.network_name + ".h5"))
            # print "evaluating model"
            # da.calculate_test_acccuracies(self.network_name, self.test_data, True, True, True, segment_size=self.segment_size)

    def active_learning_round(self, model, known: dict, round: int, X_t, X_v, y_t, y_v):
        X_pool, y_pool, pool_ident = self.create_train_data(round)
        # query for uncertainty
        query_idx = self.uncertainty_sampling(model, X_pool, n_instances=10)

        x_us = X_pool[query_idx]
        y_us = y_pool[query_idx]

        if not pool_ident in known.keys():
            known[pool_ident] = np.array()

        # ignore already added entries
        for qidx in query_idx:
            if not qidx in known[pool_ident]:
                np.append(known[pool_ident], qidx)

        x_us = np.delete(x_us, known[pool_ident], axis=0)
        y_us = np.delete(y_us, known[pool_ident], axis=0)
        
        # split the new records into test and val
        r_x_t, r_y_t, r_x_v, r_y_v = self.split_train_val_data(x_us, y_us)

        # append to used / passed sets
        np.append(X_t, r_x_t)
        np.append(y_t, r_y_t)
        np.append(X_v, r_x_v)
        np.append(y_v, r_y_v)

    def uncertainty_sampling(self, model, X, n_instances: int = 1):
        """
        Uncertainty sampling query strategy. Selects the least sure instances for labelling.
        Args:
            classifier: The classifier for which the labels are to be queried.
            X: The pool of samples to query from.
            n_instances: Number of samples to be queried.
            **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
                measure function.
        Returns:
            The indices of the instances from X chosen to be labelled;
            the instances from X chosen to be labelled.
        """
        try:
            classwise_uncertainty = model.predict(X)
        except ValueError:
            return np.ones(shape=(X.shape[0], ))

        # for each point, select the maximum uncertainty
        uncertainty = 1 - np.max(classwise_uncertainty, axis=1)
        query_idx = self.multi_argmax(uncertainty, n_instances=n_instances)
        return query_idx

    def multi_argmax(self, values: np.ndarray, n_instances: int = 1):
        """
        Selects the indices of the n_instances highest values.
        Args:
            values: Contains the values to be selected from.
            n_instances: Specifies how many indices to return.
        Returns:
            The indices of the n_instances largest values.
        """
        assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility'

        max_idx = np.argpartition(-values, n_instances-1, axis=0)[:n_instances]
        return max_idx