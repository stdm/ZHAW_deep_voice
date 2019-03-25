"""
The controller to train and test the pairwise_lstm network
"""

import numpy as np

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils.logger import *
from common.utils.paths import *
from common.utils.pickler import load
from .bilstm_2layer_dropout_plus_2dense import bilstm_2layer_dropout
from .core.data_gen import generate_test_data


class LSTMController(NetworkController):
    def __init__(self, out_layer, seg_size, vec_size):
        super().__init__("pairwise_lstm", "speakers_40_clustering_vs_reynolds")
        self.network_file = self.name + "_100"
        self.out_layer = out_layer
        self.seg_size = seg_size
        self.vec_size = vec_size

    def train_network(self):
        bilstm_2layer_dropout(
            self.network_file,
            'speakers_100_50w_50m_not_reynolds_cluster',
            n_hidden1=256,
            n_hidden2=256,
            n_classes=100,
            n_10_batches=1000,
            m=0.5,
            s=64,
            segment_size=self.seg_size
        )
