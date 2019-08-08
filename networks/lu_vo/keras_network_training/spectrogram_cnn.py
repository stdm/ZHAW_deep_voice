import keras
import numpy as np
import pickle
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Conv2D, Flatten, BatchNormalization, MaxPooling2D

from common.clustering.generate_embeddings import generate_embeddings
from common.utils.paths import *
from common.utils.logger import *
from networks.pairwise_lstm.core import data_gen as dg
from networks.pairwise_lstm.core import plot_saver as ps
import common.spectrogram.speaker_train_splitter as sts
from common.utils.ShortUtteranceConverter import create_data_lists
from common.utils import TimeCalculator


class SpectrogramCnn:

    def __init__(self, name, net_path, config):
        super().__init__()
        self.network_name = name
        self.logger = get_logger("luvo", logging.INFO)
        self.net_path = net_path
        self.config = config

    def create_net(self, channel, n_classes):
        model = Sequential()

        #convolution layer 1
        model.add(Conv2D(32, kernel_size=(4,4), activation='relu', input_shape=(channel, self.config.getint('luvo', 'seg_size'),self.config.getint('luvo', 'spectrogram_height')), data_format='channels_first'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=4,strides=2))

        #convolution layer 2
        model.add(Conv2D(64, kernel_size=(4, 4), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=4, strides=2))

        #dense layer
        model.add(Flatten())
        model.add(Dense((n_classes*10)))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.5))
        model.add(Dense((n_classes*5)))

        #output layer
        model.add(Dense((n_classes), activation='softmax'))

        sgd = keras.optimizers.SGD(lr=self.config.getfloat('luvo', 'update_learning_rate'), momentum=self.config.getfloat('luvo', 'update_momentum'), decay=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        print( model.summary())

        return model

    def create_and_train(self, training_data):
        # Load training data
        X_t, y_t, X_v, y_v = self.create_train_data(training_data)
        print(X_t.shape)

        # Create network
        model = self.create_net(X_t.shape[1], self.config.getint('luvo','n_classes'))

        # Set new batch iterator
        train_gen = dg.batch_generator(X_t, y_t, 128, segment_size=self.config.getint('luvo', 'seg_size'))
        val_gen = dg.batch_generator(X_v, y_v, 128, segment_size=self.config.getint('luvo', 'seg_size'))

        # Train the network
        self.logger.info("Fitting...")
        history = model.fit_generator(train_gen, steps_per_epoch=10, epochs=self.config.getint('luvo', 'num_epochs'),
                                      verbose=2, callbacks=None, validation_data=val_gen,
                                      validation_steps=2, class_weight=None, max_q_size=10,
                                      nb_worker=1, pickle_safe=False)
        ps.save_accuracy_plot(history, self.network_name)
        ps.save_loss_plot(history, self.network_name)
        print("saving model")
        model.save(get_experiment_nets(self.network_name + ".h5"))

    def create_train_data(self, training_data):
        with open(get_speaker_pickle(training_data), 'rb') as f:
            (X, y, speaker_names) = pickle.load(f)

            splitter = sts.SpeakerTrainSplit(0.2)
            X_t, X_v, y_t, y_v = splitter(X, y)

        return X_t, y_t, X_v, y_v

    def create_embeddings(self, X_train, y_train, X_test, y_test):
        seg_size = self.config.getint('luvo', 'seg_size')
        short_utterance = self.config.getboolean('test', 'short_utterances')

        x_train, speakers_train = prepare_data(X_train, y_train, seg_size)
        x_test, speakers_test = prepare_data(X_test, y_test, seg_size)

        x_list, y_list, _ = create_data_lists(short_utterance, x_train, x_test, speakers_train, speakers_test)

        # Load the network and add Batchiterator
        network_file = get_experiment_nets(self.network_name + ".h5")
        model_full = load_model(network_file)
        model_full.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        # Get a Model with the embedding layer as output and predict
        model_partial = Model(inputs=model_full.input, outputs=model_full.layers[self.config.getint('luvo', 'out_layer')].output)

        x_cluster_list = []
        y_cluster_list = []
        for x_data, y_data in zip(x_list, y_list):
            print(x_data.shape)
            x_cluster = np.asarray(model_partial.predict(x_data))
            x_cluster_list.append(x_cluster)
            y_cluster_list.append(y_data)


        embeddings, speakers, num_embeddings = generate_embeddings(x_cluster_list, y_cluster_list, x_cluster_list[0].shape[1])

        # Calculate the time per utterance
        time = TimeCalculator.calc_time_all_utterances(y_cluster_list, self.config.getint('luvo', 'seg_size'))

        return embeddings, speakers, num_embeddings, time

def prepare_data(X,y, segment_size):
    x, speakers = dg.generate_test_data(X, y, segment_size)

    # Reshape test data because it is an lstm
    return x.reshape(x.shape[0], 1, x.shape[3], x.shape[2]), speakers
