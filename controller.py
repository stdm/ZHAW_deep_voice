"""
The main entry point of the speaker clustering suite.
You can use this file to setup, train and test any network provided in the suite.
Note that not all networks use all of the available parameters. Check their implementation beforehand.

Usage: controller.py [-h] [-setup] [-n NETWORKS [NETWORKS ...]] [-train] [-test] [-clear]

Controller suite for Speaker clustering

optional arguments:
  -h, --help         show this help message and exit
  -setup             Run project setup.
  -n                 The networks to use for training or analysis.
                     Available: 'pairwise_lstm', 'pairwise_lstm_vox2', 'arc_face', 'pairwise_kldiv', 'luvo', 'gmm', 'i-vector'.
                     All networks use different sets of parameters which can be configured in the file common/config.cfg
  -train             Train the specified network.
  -test              Test the specified network.
  -best              Just the best results of the networks will be used in -train or -plot
  -plot              Plots the last results of the specified networks in one file.
  -dev               Enable dev mode so the dev set instead of the test set is used for testing
"""


import argparse
import sys

from common.analysis.plotting import plot_files
from common.extrapolation.setup import setup_suite, is_suite_setup
from common.utils.paths import *
from common.utils.load_config import *


# Constants
DEFAULT_SETUP = False
DEFAULT_NETWORKS = ()
DEFAULT_TRAIN = False
DEFAULT_TEST = False
DEFAULT_PLOT = False
DEFAULT_BEST = False
DEFAULT_DEV = False
DEFAULT_CONFIG = load_config(None, join(get_configs(), 'config.cfg'))


class Controller:
    def __init__(self, config=DEFAULT_CONFIG,
                 setup=DEFAULT_SETUP, networks=DEFAULT_NETWORKS, train=DEFAULT_TRAIN, test=DEFAULT_TEST,
                 plot=DEFAULT_PLOT, best=DEFAULT_BEST, dev=DEFAULT_DEV):
        self.config = config
        self.setup = setup
        self.networks = networks
        self.train = train
        self.test = test
        self.plot = plot
        self.best = best
        self.dev = dev
        self.network_controllers = []

    def setup_networks(self):
        if is_suite_setup():
            print("Already fully setup.")
        else:
            print("Setting up the network suite.")
            setup_suite()

    def train_network(self):
        for network_controller in self.network_controllers:
            network_controller.train_network()

    def test_network(self):
        for network_controller in self.network_controllers:
            network_controller.test_network()

    def plot_results(self):
        for network_controller in self.network_controllers:
            name = network_controller.name
            plot_files(name, get_result_files(name, self.best))

        if len(self.network_controllers) > 1:
            plot_files('all', get_result_files('', self.best))

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
        for network in self.networks:
            if network == 'pairwise_lstm':
                from networks.pairwise_lstm.lstm_controller import LSTMController
                self.network_controllers.append(LSTMController(self.config, self.dev, self.best))
            elif network == 'pairwise_kldiv':
                from networks.pairwise_kldiv.kldiv_controller import KLDivController
                self.network_controllers.append(KLDivController(self.config, self.dev))
            elif network == 'i_vector':
                from networks.i_vector.ivec_controller import IVECController
                self.network_controllers.append(IVECController(self.config, self.dev))
            elif network == 'luvo':
                from networks.lu_vo.luvo_controller import LuvoController
                self.network_controllers.append(LuvoController(self.config, self.dev))
            elif network == 'gmm':
                from networks.gmm.gmm_controller import GMMController
                self.network_controllers.append(GMMController(self.config, self.dev))
            elif network == 'pairwise_lstm_vox2':
                from networks.pairwise_lstm_vox.lstm_controller import LSTMVOX2Controller
                self.network_controllers.append(LSTMVOX2Controller(self.config, self.dev, self.best))
            else:
                print("Network " + network + " is not known.")
                sys.exit(1)


if __name__ == '__main__':
    # Parse console Args
    parser = argparse.ArgumentParser(description='Controller suite for Speaker clustering')
    # add all arguments and provide descriptions for them
    parser.add_argument('-setup', dest='setup', action='store_true',
                        help='Run project setup.')
    parser.add_argument('-n', nargs='+', dest='networks', default=DEFAULT_NETWORKS,
                        help='The networks to use for training or analysis.')
    parser.add_argument('-train', dest='train', action='store_true',
                        help='Train the specified network.')
    parser.add_argument('-test', dest='test', action='store_true',
                        help='Test the specified network.')
    parser.add_argument('-plot', dest='plot', action='store_true',
                        help='Plots the results of the specified networks.')
    parser.add_argument('-best', dest='best', action='store_true',
                        help='If a single Network is specified and plot was called, just the best curves will be plotted. If test was called only the best network will be tested')
    parser.add_argument('-dev', dest='dev', action='store_true',
                        help='Enable dev mode so the dev set instead of the test set is used for testing')

    args = parser.parse_args()

    controller = Controller(
        setup=args.setup, networks=tuple(args.networks), train=args.train, test=args.test, plot=args.plot,
        best=args.best, dev=args.dev
    )

    controller.run()
