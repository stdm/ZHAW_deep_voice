"""
The main entry point of the speaker clustering suite.
You can use this file to setup, train and test any network provided in the suite.
Note that not all networks use all of the available parameters. Check their implementation beforehand.

Usage: controller.py [-h] [-setup] [-n NETWORK] [-train] [-test] [-clear]

Controller suite for Speaker clustering

optional arguments:
  -h, --help         show this help message and exit
  -setup             Run project setup.
  -n NETWORK         The network to use for training or analysis.
  -train             Train the specified network.
  -test              Test the specified network.
  -clear             Clean directories before starting network.
  -debug             Set loglevel for TensorFlow and Logger to debug.
  -best              Just the best results of the networks will be used in -train or -plot
  -plot              Plots the last results of the specified networks in one file.
  -seg_size          The segment size used during training and testing
  -vec_size          The vector size to be expected when choosing a certain out_layer
  -out_layer         The layer index of the network to use for testing / creating the embeddings
  -epoch_total       number of epochs to train the network
  -epochs_pre_alr    number of epochs to train before active learning process starts
  -alr               number of active learning rounds (can be 0)
  -dense_factor      width of certain dense layers can be controlled via this parameter
  -output_size       width of the last dense layer (should be >= dense_factor)

"""


import argparse
import sys

import matplotlib

from networks.lstm_arc_face.arc_face_controller import ArcFaceController

matplotlib.use('Agg')

from common.analysis.analysis import *
from common.extrapolation.setup import setup_suite, is_suite_setup
from common.utils.paths import *
from common.utils.load_config import *

# Controllers
# -------------------
from networks.lu_vo.luvo_controller import LuvoController
from networks.pairwise_kldiv.kldiv_controller import KLDivController
from networks.pairwise_lstm.lstm_controller import LSTMController
from networks.pairwise_lstm_vox.lstm_controller import LSTMVOX2Controller
from networks.gmm.gmm_controller import GMMController
from networks.i_vector.ivec_controller import IVECController


# Constants
# -------------------
DEFAULT_SETUP = False
DEFAULT_NETWORK = 'pairwise_lstm_vox2'
DEFAULT_TRAIN = False
DEFAULT_TEST = False
DEFAULT_PLOT = False
DEFAULT_BEST = False
DEFAULT_DEV = False
DEFAULT_CONFIG = load_config(None, join(get_common(), 'config.cfg'))


class Controller:
    def __init__(self, config=DEFAULT_CONFIG,
                 setup=DEFAULT_SETUP, network=DEFAULT_NETWORK, train=DEFAULT_TRAIN, test=DEFAULT_TEST,
                plot=DEFAULT_PLOT, best=DEFAULT_BEST, dev=DEFAULT_DEV):
        self.setup = setup
        self.network = network
        self.train = train
        self.test = test
        self.network_controllers = []
        self.plot = plot
        self.best = best
        self.dev = dev
        self.config = config

    def train_network(self):
        for network_controller in self.network_controllers:
            network_controller.train_network()

    def test_network(self):
        for network_controller in self.network_controllers:
            network_controller.test_network()

    def plot_results(self):
        for network_controller in self.network_controllers:
            nn = network_controller.get_formatted_result_network_name()
            plot_files(nn, self._get_result_files())



    def _get_result_files(self, filename):
        if self.network == "all":
            regex = '^.*best\.pickle'
        elif self.best:
            regex = '^{}.*best.pickle'.format(filename)
        else:
            regex = '^{}.*(?<!best)\.pickle'.format(filename)

        files = list_all_files(get_results(), regex)

        for index, file in enumerate(files):
            files[index] = get_results(file)
        return files

    def get_embeddings(self):
        return None, None, None, None

    def run(self):

        # Setup
        if self.setup:
            self.setup_networks()

        # Validate network
        self.generate_controllers()

        # Train network
        if self.train:
            self.train_network()

        # Test network
        if self.test:
            self.test_network()


        # Plot results
        if self.plot:
            self.plot_results()

    def generate_controllers(self):

        controller_dict = {
            'pairwise_lstm': [LSTMController(self.config, self.dev, self.best)],
            'pairwise_kldiv': [KLDivController(self.config, self.dev)],
            'i_vector': [IVECController(self.config, self.dev)],
            'luvo': [LuvoController(self.config, self.dev)],
            'gmm': [GMMController(self.config, self.dev)],
            'all': [LSTMController(self.config, self.dev, self.best), KLDivController(self.config, self.dev),
                    LuvoController(self.config, self.dev)],
            'pairwise_lstm_vox2': [LSTMVOX2Controller(self.config, self.dev)],
            'arc_face': [ArcFaceController(self.config, self.dev)]
        }

        try:
            self.network_controllers = controller_dict[self.network]
        except KeyError:
            print("Network " + self.network + " is not known:")
            print("Valid Names: ", join([k for k in controller_dict.keys()]))
            sys.exit(1)


    def setup_networks(self):
        if is_suite_setup():
            print("Already fully setup.")
        else:
            print("Setting up the network suite.")
            setup_suite()


    def get_file_format(self):
        if self.network == 'pairwise_lstm_vox2':
            return '.h5'
        else:
            return '.pickle'


if __name__ == '__main__':
    # Parse console Args
    parser = argparse.ArgumentParser(description='Controller suite for Speaker clustering')
    # add all arguments and provide descriptions for them
    parser.add_argument('-setup', dest='setup', action='store_true',
                        help='Run project setup.')
    parser.add_argument('-n', dest='network', default=DEFAULT_NETWORK,
                        help='The network to use for training or analysis.')
    parser.add_argument('-train', dest='train', action='store_true',
                        help='Train the specified network.')
    parser.add_argument('-test', dest='test', action='store_true',
                        help='Test the specified network.')
    parser.add_argument('-plot', dest='plot', action='store_true',
                        help='Plots the last results of the specified networks in one file.')
    parser.add_argument('-best', dest='best', action='store_true',
                        help='If a single Network is specified and plot was called, just the best curves will be plotted. If test was called only the best network will be tested')
    parser.add_argument('-dev', dest='dev', action='store_true',
                        help='If this flag is true, the dev-set is used for testing the network')

    args = parser.parse_args()
    config = load_config(None, join(get_common(), 'config.cfg'))

    controller = Controller(
        setup=args.setup, network=args.network, train=args.train, test=args.test, plot=args.plot, best=args.best, dev=args.dev
    )

    controller.run()
