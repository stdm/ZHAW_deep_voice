"""
The main entry point of the speaker clustering suite.
You can use this file to setup, train and test any network provided in the suite.

Usage: controller.py [-h] [-setup] [-n NETWORK] [-train] [-test] [-clear]
                     [-debug]

Controller suite for Speaker clustering

optional arguments:
  -h, --help  show this help message and exit
  -setup      Run project setup.
  -n NETWORK  The network to use for training or analysis.
  -train      Train the specified network.
  -test       Test the specified network.
  -clear      Clean directories before starting network.
  -debug      Set loglevel for TensorFlow and Logger to debug.
  -plot       Plots the last results of the specified networks in one file.

"""
import argparse
import sys

import matplotlib

matplotlib.use('Agg')
from common.analysis.analysis import *
from common.extrapolation.setup import setup_suite, is_suite_setup
from common.network_controller import NetworkController
from common.utils.paths import *
from networks.flow_me.me_controller import MEController
from networks.lu_vo.luvo_controller import LuvoController
from networks.pairwise_kldiv.kldiv_controller import KLDivController
from networks.pairwise_lstm.lstm_controller import LSTMController


class Controller(NetworkController):
    def __init__(self, setup=True, network='pairwise_lstm', train=False, test=False, clear=False, debug=False,
                 plot=False, best=False, val_data=40):
        super().__init__("Front")
        self.setup = setup
        self.network = network
        self.train = train
        self.test = test
        self.clear = clear
        self.debug = debug
        self.network_controllers = []
        self.plot = plot
        self.best = best

        validation_data = {
            40: "speakers_40_clustering_vs_reynolds",
            60: "speakers_60_clustering",
            80: "speakers_80_clustering"
        }

        self.val_data = validation_data[val_data]

    def train_network(self):
        if not self.train:
            return

        for network_controller in self.network_controllers:
            network_controller.train_network()

    def test_network(self):
        if not self.test:
            return

        for network_controller in self.network_controllers:
            network_controller.test_network()

    def get_embeddings(self):
        return None, None, None, None

    def run(self):

        # Setup
        self.setup_networks()

        # Validate network
        self.generate_controllers()

        # Train network
        self.train_network()

        # Test network
        self.test_network()

        # Plot results
        self.plot_results()

    def generate_controllers(self):

        controller_dict = {
            'pairwise_lstm': [LSTMController()],
            'pairwise_kldiv': [KLDivController()],
            'flow_me': [MEController(self.clear, self.debug, False)],
            'luvo': [LuvoController()],
            'all': [LSTMController(), KLDivController(), MEController(self.clear, self.debug, False), LuvoController()]
        }

        try:
            self.network_controllers = controller_dict[self.network]
            for net in self.network_controllers:
                net.val_data = self.val_data

        except KeyError:
            print("Network " + self.network + " is not known:")
            print("Valid Names: ", join([k for k in controller_dict.keys()]))
            sys.exit(1)

    def setup_networks(self):
        if not self.setup:
            return

        if is_suite_setup():
            print("Already fully setup.")
        else:
            print("Setting up the network suite.")

        setup_suite()

    def plot_results(self):
        if not self.plot:
            return

        plot_files(self.network, self.get_result_files())

    def get_result_files(self):
        if self.network == "all":
            regex = '*best*.pickle'
        elif self.best:
            regex = self.network + '*best*.pickle'
        else:
            regex = self.network + ".pickle"

        files = list_all_files(get_results(), regex)
        for index, file in enumerate(files):
            files[index] = get_results(file)
        return files

if __name__ == '__main__':
    # Parse console Args
    parser = argparse.ArgumentParser(description='Controller suite for Speaker clustering')
    parser.add_argument('-setup', dest='setup', action='store_true', help='Run project setup.')
    parser.add_argument('-n', dest='network', default='pairwise_lstm',
                        help='The network to use for training or analysis.')
    parser.add_argument('-train', dest='train', action='store_true', help='Train the specified network.')
    parser.add_argument('-test', dest='test', action='store_true', help='Test the specified network.')
    parser.add_argument('-clear', dest='clear', action='store_true', help='Clean directories before starting network.')
    parser.add_argument('-debug', dest='debug', action='store_true',
                        help='Set loglevel for TensorFlow and Logger to debug')
    parser.add_argument('-plot', dest='plot', action='store_true',
                        help='Plots the last results of the specified networks in one file.')
    parser.add_argument('-best', dest='best', action='store_true',
                        help='If a single Network is specified and plot was called, just the best curves will be plotted')
    parser.add_argument('-val#', dest='validation_number', default=40,
                        help='Specify how many speakers should be used for testing (40, 60, 80).')
    args = parser.parse_args()

    controller = Controller(args.setup, args.network, args.train, args.test, args.clear, args.debug, args.plot, args.best, args.validation_number)
    controller.run()
