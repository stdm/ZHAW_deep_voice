import numpy as np
np.random.seed(1337)  # for reproducibility

from networks.pairwise_lstm_vox.core import plot_saver as ps
from common.spectrogram import speaker_train_splitter as sts
from common.utils.pickler import load_speaker_pickle_or_h5
from common.utils.paths import *

'''This Class performs active learning rounds
    
    Parameters:
    # logger: The logger used to pass messages
    # n_instances: Amount of speaker samples extracted from a given pool during one active learning round
    # epochs: Number of epochs to train the network in total
    # epochs_per_round: Amount of epochs to be run per active learning round
    # segment_size: The segment length used for training
    # training_data: An identifier to load the training dataset

    Work of Lauener and Lehmann.
'''

class active_learner(object):
    def __init__(self, logger, segment_size, epochs, training_data, epochs_per_round, n_instances: int = 128):
        self.logger = logger
        self.n_instances = n_instances
        self.epochs_per_round = epochs_per_round
        self.epochs = epochs
        self.training_data = training_data
        self.segment_size = segment_size

        # temporary variable used to check if additional pools need to be evaluated or
        # if the maximum number has been found according to the file name schema
        self.active_learning_pools_increment_active = True
        
        # Temporary variable that holds the amount of active learning pools found, used to
        # keep cycling across the full pool with modulo
        self.active_learning_pools = 0
        
    # This methods performs the active learning rounds and needs the following parameters:
    # 
    # model: The current model that is used to train
    # network: The network that implements the :fit method
    # callbacks: The callbacks used by the network during the :run_network "pre active learning" training
    # epochs_trained: The amount of epochs already run
    # active_learning_rounds: How many rounds of active learning are performed
    # X_t, y_t: The features and labels of the training set
    # X_v, y_v: The features and labels of the validation set
    # 
    # Returns: The model used in training
    # 
    def perform_active_learning(self, model, network, callbacks, epochs_trained, active_learning_rounds, X_t, y_t, X_v, y_v):
        known_pool_data = dict()

        X_t_shapes = [ X_t.shape[0] ]
        X_v_shapes = [ X_v.shape[0] ]

        for i in range(active_learning_rounds):
            self.logger.info("Active learning round " + str(i) + "/" + str(active_learning_rounds))

            if self.epochs >= (epochs_trained + self.epochs_per_round):
                epochs_to_run = self.epochs_per_round
            else:
                epochs_to_run = self.epochs - epochs_trained

            # if max epochs to train already reached before all rounds processed we can end the training
            if epochs_to_run <= 0:
                self.logger.info("Max epoch of " + str(self.epochs) + " reached, end of training")
                break

            # query for uncertainty based on pool and append to numpy X_t, X_v, ... arrays
            (X_t, X_v, y_t, y_v) = self.perform_round(model, known_pool_data, i, X_t, X_v, y_t, y_v)
            
            network.fit(model, callbacks, X_t, X_v, y_t, y_v, epochs_to_run)

            epochs_trained += epochs_to_run

            X_t_shapes.append(X_t.shape[0])
            X_v_shapes.append(X_v.shape[0])

            ps.save_alr_shape_x_plot(network.network_name, [ X_t_shapes, X_v_shapes ])

        return model

    # This method performs ONE round of active learning by itself
    # 
    # The parameters are:
    # model: The model used for training
    # known_pool_data: A dictionary containing the information which samples have been used already (to avoid duplicates)
    # current_round: Which active learning round is performed
    # X_t, y_t: The features and labels of the training set
    # X_v, y_v: The features and labels of the validation set
    # 
    # Returns: The new training and validation sets, with new smaples added from the speaker pools
    # 
    def perform_round(self, model, known_pool_data: dict, current_round: int, X_t, X_v, y_t, y_v):
        X_pool, y_pool, pool_ident = self._reader_speaker_data_round(current_round)

        # query for uncertainty
        query_idx = self.perform_uncertainty_sampling(model, X_pool, self.n_instances)
        
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
        r_x_t, r_y_t, r_x_v, r_y_v = self._split_train_val_data(x_us, y_us)
        
        # append to used / passed sets
        new_X_t = np.append(X_t, r_x_t, axis=0)
        new_X_v = np.append(X_v, r_x_v, axis=0)
        new_y_t = np.append(y_t, r_y_t, axis=0)
        new_y_v = np.append(y_v, r_y_v, axis=0)
        
        return new_X_t, new_X_v, new_y_t, new_y_v

    # Uncertainty sampling query strategy. Selects the least sure instances for labelling.
    #
    # Parameters:
    # model: The model for which the labels are to be queried.
    # X: The pool of samples to query from.
    # n_instances: Number of samples to be queried.
    # 
    # Returns: The indices of the instances from X chosen to be labelled;
    #
    def perform_uncertainty_sampling(self, model, X, n_instances: int = 1):
        try:
            # lehmacl1@2019-05-01: Currently takes only the first :segment_size ms to evaluate
            # the classwise uncertainty for comparison
            #
            reshaped_X = X.reshape(X.shape[0], X.shape[3], X.shape[2])
            resized_X = reshaped_X[:, range(self.segment_size), :]

            classwise_uncertainty = model.predict(resized_X)
        except ValueError:
            classwise_uncertainty = np.ones(shape=(X.shape[0], ))

        # for each point, select the maximum uncertainty
        uncertainty = 1 - np.max(classwise_uncertainty, axis=1)
        query_idx = self._multi_argmax(uncertainty, n_instances=n_instances)
        return query_idx

    # Selects the indices of the n_instances highest values ("best samples to train on")
    # 
    # Paramters:
    # values: Contains the values to be selected from
    # n_instances: Specifies how many indicies to return (top N)
    # 
    # Returns: The :n_instances indices with the largest values
    # 
    def _multi_argmax(self, values: np.ndarray, n_instances: int = 1):
        assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility'
        max_idx = np.argpartition(-values, n_instances-1, axis=0)[:n_instances]

        return max_idx

    # Reads the speaker data from a given speaker pickle or h5
    # 
    # Paramters:
    # speaker_pickle: Location of speaker pickle/h5 file
    #
    # Returns: The unpacked file (Features, Labels, Speakerident List)
    #
    def _read_speaker_data(self, speaker_pickle):
        self.logger.info('create_train_data ' + speaker_pickle)
        (X, y, _) = load_speaker_pickle_or_h5(speaker_pickle)
        return (X, y, speaker_pickle)

    def _reader_speaker_data_round(self, al_round, recurse=0):
        if recurse >= 2:
            raise Exception("Recursion was not applied correctly in active learning speaker pool detection!")

        training_data_round = self.training_data + '_' + str(al_round)
        speaker_pickle = get_speaker_pickle(training_data_round, format='.h5')

        if not path.exists(speaker_pickle):
            self.active_learning_pools_increment_active = False

            return self._reader_speaker_data_round(al_round=al_round % self.active_learning_pools, recurse=recurse+1)
        else:
            if self.active_learning_pools_increment_active:
                self.active_learning_pools += 1
            
            self.logger.info('create_train_data ' + self.training_data + ' pool ' + training_data_round)
            return self._read_speaker_data(speaker_pickle)

    # Performs a 80-20 split on the given Features and Labels
    # 
    def _split_train_val_data(self, X, y):
        splitter = sts.SpeakerTrainSplit(0.2)
        X_t, X_v, y_t, y_v = splitter(X, y)
        return X_t, y_t, X_v, y_v