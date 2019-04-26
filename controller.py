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
import sys

import matplotlib
matplotlib.use('Agg')

from common.analysis.analysis import *
from common.extrapolation.setup import setup_suite, is_suite_setup
from common.utils.paths import *
from common.utils.load_config import *

# Controllers
# -------------------
from networks.flow_me.me_controller import MEController
from networks.lu_vo.luvo_controller import LuvoController
from networks.pairwise_kldiv.kldiv_controller import KLDivController
from networks.pairwise_lstm.lstm_controller import LSTMController
from networks.i_vector.ivec_controller import IVECController


class Controller:
    def __init__(self, config):
        self.setup = config.getboolean('common', 'setup')
        self.network = config.get('common', 'network')
        self.train = config.getboolean('common', 'train')
        self.test = config.getboolean('common', 'test')
        self.clear = config.getboolean('common', 'clear')
        self.debug = config.getboolean('common', 'debug')
        self.network_controllers = []
        self.plot = config.getboolean('common', 'plot')
        self.best = config.getboolean('common', 'best')
        self.config = config


    def train_network(self):
        for network_controller in self.network_controllers:
            network_controller.train_network()

    def test_network(self):
        for network_controller in self.network_controllers:
            network_controller.test_network()

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
            'pairwise_lstm': [LSTMController(self.config)],
            'pairwise_kldiv': [KLDivController(self.config)],
            #'flow_me': [MEController(self.clear, self.debug, False)],
            'luvo': [LuvoController(self.config)],
            'ivector': [IVECController(self.config)],
            'all': [LSTMController(self.config), KLDivController(self.config), LuvoController(self.config)]
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

    def plot_results(self):
        plot_files(self.network, self._get_result_files())

    def _get_result_files(self):
        if self.network == "all":
            regex = '^.*best\.pickle'
        elif self.best:
            regex = '^{}.*best.pickle'.format(self.network)
        else:
            regex = '^{}.*(?<!best)\.pickle'.format(self.network)

        files = list_all_files(get_results(), regex)

        for index, file in enumerate(files):
            files[index] = get_results(file)
        return files

if __name__ == '__main__':
    config = load_config(None, join(get_common(), 'config.cfg'))
    controller = Controller(config)
    controller.run()
