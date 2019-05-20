"""
A NetworkController contains all knowledge and links to successfully train and test a network.

It's used to encapsulate a network and make use of shared code.
"""
import abc

from common.analysis.analysis import analyse_results
from common.clustering.generate_clusters import cluster_embeddings
from common.utils.paths import get_speaker_pickle


class NetworkController:
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, config):
        self.val_data = config.get('validation', 'test_pickle')
        self.dev_val_data = config.get('validation', 'dev_pickle')
        self.name = name
        self.dev_mode = config.getboolean('validation', 'dev_mode')
        self.config = config

    def get_validation_train_data(self):
        if self.dev_mode:
            return get_speaker_pickle(self.dev_val_data + "_train")
        else:
            return get_speaker_pickle(self.val_data + "_train")

    def get_validation_test_data(self):
        if self.dev_mode:
            return get_speaker_pickle(self.dev_val_data + "_test")
        else:
            return get_speaker_pickle(self.val_data + "_test")

    def get_validation_data_name(self):
        if self.dev_mode:
            return self.dev_val_data
        else:
            return self.val_data


    @abc.abstractmethod
    def train_network(self):
        """
        This method implements the training/fitting of the neural netowrk this controller implements.
        It handles the cycles, logging and saving.
        :return:
        """
        pass

    @abc.abstractmethod
    def get_embeddings(self):
        """
        Processes the validation list and get's the embeddings as the network output.
        All return values are sets of possible multiples.
        :return: checkpoints, embeddings, speakers, speaker numbers and the time per utterance
        """
        return None, None, None, None, None

    def get_clusters(self):
        """
        Generates the predicted_clusters with the results of get_embeddings.
        All return values are sets of possible multiples.
        :return:
        checkpoint_names: A list of names from the checkpoints. Later used as curvenames,
        set_of_predicted_clusters: A 2D array of the predicted Clusters from the Network. [checkpoint, clusters]
        set_of_true_clusters: A 2d array of the validation clusters. [checkpoint, validation-clusters]
        embeddings_numbers: A list which represent the number of embeddings in each checkpoint.
        set_of_times: A 2D array containing the time per utterance
        """
        checkpoint_names, set_of_embeddings, set_of_true_clusters, embeddings_numbers, set_of_times =\
            self.get_embeddings()
        set_of_predicted_clusters = cluster_embeddings(set_of_embeddings, set_of_true_clusters,
                                                       self.config.getboolean('validation', 'dominant_set'))

        return checkpoint_names, set_of_predicted_clusters, set_of_true_clusters, embeddings_numbers, set_of_times

    def test_network(self):
        """
        Tests the network implementation with the validation data set and saves the result sets
        of the different metrics in analysis.
        """
        checkpoint_names, set_of_predicted_clusters, set_of_true_clusters, embeddings_numbers, set_of_times =\
            self.get_clusters();
        network_name = self.name + '_' + self.val_data
        analyse_results(network_name, checkpoint_names, set_of_predicted_clusters, set_of_true_clusters,
                        embeddings_numbers, set_of_times)
