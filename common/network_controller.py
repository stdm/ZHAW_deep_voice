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

    def __init__(self, name, val_data):
        self.name = name
        self.val_data = val_data

    def get_network_name(self):
        return self.name + '_' + self.val_data

    def get_validation_train_data(self):
        return get_speaker_pickle(self.val_data + "_train")

    def get_validation_test_data(self):
        return get_speaker_pickle(self.val_data + "_test")

    @abc.abstractmethod
    def train_network(self):
        """
        This method implements the training/fitting of the neural netowrk this controller implements.
        It handles the cycles, logging and saving.
        :return:
        """
        pass

    @abc.abstractmethod
    def get_embeddings(self, out_layer, seg_size, vec_size):
        """
        Processes the validation list and get's the embeddings as the network output.
        All return values are sets of possible multiples.
        :return: checkpoints, embeddings, speakers and the speaker numbers
        """
        return None, None, None, None

    def get_clusters(self, out_layer, seg_size, vec_size):
        """
        Generates the predicted_clusters with the results of get_embeddings.
        All return values are sets of possible multiples.
        :return:
        checkpoint_names: A list of names from the checkpoints. Later used as curvenames,
        set_of_predicted_clusters: A 2D array of the predicted Clusters from the Network. [checkpoint, clusters]
        set_of_true_clusters: A 2d array of the validation clusters. [checkpoint, validation-clusters]
        embeddings_numbers: A list which represent the number of embeddings in each checkpoint.
        """
        checkpoint_names, set_of_embeddings, set_of_true_clusters, embeddings_numbers = self.get_embeddings(out_layer=out_layer, seg_size=seg_size, vec_size=vec_size)
        set_of_predicted_clusters = cluster_embeddings(set_of_embeddings)

        return checkpoint_names, set_of_predicted_clusters, set_of_true_clusters, embeddings_numbers

    def test_network(self, out_layer, seg_size, vec_size):
        """
        Tests the network implementation with the validation data set and saves the result sets
        of the different metrics in analysis.
        """
        network_name = self.get_network_name()
        checkpoint_names, set_of_predicted_clusters, set_of_true_clusters, embeddings_numbers = self.get_clusters(out_layer, seg_size, vec_size)
        
        analyse_results(network_name, checkpoint_names, set_of_predicted_clusters, set_of_true_clusters,
                        embeddings_numbers)
