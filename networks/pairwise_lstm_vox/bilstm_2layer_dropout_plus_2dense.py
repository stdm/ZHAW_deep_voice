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

from .core import data_gen as dg
from .core import pairwise_kl_divergence as kld
from .core import plot_saver as ps
from .core.plot_callback import PlotCallback

import common.spectogram.speaker_train_splitter as sts
from common.utils.paths import *
from common.utils.pickler import load_speaker_pickle_or_h5

'''This Class Trains a Bidirectional LSTM with 2 Layers, and 2 Denselayer and a Dropout Layers
    Parameters:
    name: Network/experiment name to save artifacts
    training_data: name of the Training data file, expected to be train_xxxx.pickle
    n_hidden1: Units of the first LSTM Layer
    n_hidden2: Units of the second LSTM Layer
    dense_factor: Amount of output classes (Speakers in Trainingset)
    epochs: Number of Epochs to train the Network per ActiveLearningRound
    activeLearnerRounds: Number of learning rounds to requery the pool for new data
    segment_size: Segment size that is used as input 100 equals 1 second with current Spectrogram extraction
    frequency: size of the frequency Dimension of the Input Spectrogram

    Work of Gerber and Glinski.
'''


class bilstm_2layer_dropout(object):
    def __init__(self, name, training_data, n_hidden1, n_hidden2, dense_factor, 
                 epochs, active_learning_rounds,
                 segment_size, frequency=128):
        self.network_name = name
        self.training_data = training_data
        self.test_data = 'test' + training_data[5:]
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.dense_factor = dense_factor
        self.epochs = epochs
        self.active_learning_rounds = active_learning_rounds
        self.active_learning_pools = 0
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

        model.add(Dense(self.dense_factor * 10))
        model.add(Dropout(0.25))
        model.add(Dense(self.dense_factor * 5))
        model.add(Dense(self.dense_factor))
        model.add(Activation('softmax'))
        adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        model.compile(loss=kld.pairwise_kl_divergence, optimizer=adam, metrics=['accuracy'])
        print(model)
        return model

    def read_speaker_data(self, speaker_pickle):
        print('create_train_data', speaker_pickle)
        (X, y, _) = load_speaker_pickle_or_h5(speaker_pickle)

        return (X, y, speaker_pickle)

    def reader_speaker_data_round(self, al_round):
        training_data_round = self.training_data + '_' + str(al_round)
        speaker_pickle = get_speaker_pickle(training_data_round, format='.h5')

        if not path.exists(speaker_pickle):
            return self.reader_speaker_data_round(al_round % self.active_learning_pools)
        else:
            print('create_train_data', self.training_data, 'pool', training_data_round)
            self.active_learning_pools += 1
            return self.read_speaker_data(speaker_pickle)

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
        plott_callback_instance = PlotCallback(self.network_name)

        return [csv_logger, net_saver, plott_callback_instance]

    def create_round_specific_callbacks(self, global_callbacks, al_round):
        net_checkpoint = keras.callbacks.ModelCheckpoint(
            get_experiment_nets(self.network_name + "_" + str(al_round) + "_{epoch:05d}.h5"), 
            period=self.epochs
        )

        return global_callbacks + [net_checkpoint]

    def run_network(self):
        # base keras network
        model = self.create_net()
        global_calls = self.create_callbacks()
        known_pool_data = dict()

        # initial train set
        speaker_pickle = get_speaker_pickle(self.training_data, format='.h5')
        X_pool, y_pool, _ = self.read_speaker_data(speaker_pickle)
        X_t, y_t, X_v, y_v = self.split_train_val_data(X_pool, y_pool)
        X_t_shapes = [ X_t.shape[0] ]
        X_v_shapes = [ X_v.shape[0] ]

        for i in range(self.active_learning_rounds): # ActiveLearning Rounds
            calls = self.create_round_specific_callbacks(global_calls, i)

            if i != 0:
                # query for uncertainty based on pool and append to numpy X_t, X_v, ... arrays
                self.active_learning_round(model, known_pool_data, i, X_t, X_v, y_t, y_v)

            train_gen = dg.batch_generator_lstm(X_t, y_t, 100, segment_size=self.segment_size)
            val_gen = dg.batch_generator_lstm(X_v, y_v, 100, segment_size=self.segment_size)

            X_t_shapes.append(X_t.shape[0])
            X_v_shapes.append(X_v.shape[0])

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
            ps.save_alr_shape_x_plot(self.network_name, [ X_t_shapes, X_v_shapes ])

            print("saving model")
            model.save(get_experiment_nets(self.network_name + ".h5"))

    def active_learning_round(self, model, known_pool_data: dict, round: int, X_t, X_v, y_t, y_v):
        X_pool, y_pool, pool_ident = self.reader_speaker_data_round(round)
        # query for uncertainty
        query_idx = self.uncertainty_sampling(model, X_pool, n_instances=100)

        # Converts np.ndarray to dytpe int, default is float
        query_idx = query_idx.astype('int')

        x_us = X_pool[query_idx]
        y_us = y_pool[query_idx]

        if not pool_ident in known_pool_data.keys():
            known_pool_data[pool_ident] = []

        # ignore already added entries
        for qidx in query_idx:
            if not qidx in known_pool_data[pool_ident]:
                known_pool_data[pool_ident].append(qidx)

        numpArray = np.array(known_pool_data[pool_ident])
        x_us = np.delete(x_us, numpArray, axis=0)
        y_us = np.delete(y_us, numpArray, axis=0)
        
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
            model: The model for which the labels are to be queried.
            X: The pool of samples to query from.
            n_instances: Number of samples to be queried.
        Returns:
            The indices of the instances from X chosen to be labelled;
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
