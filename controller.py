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
from common.network_controller import NetworkController
from common.utils.paths import *
from common.utils.load_config import *

# Controllers
# -------------------
from networks.flow_me.me_controller import MEController
from networks.lu_vo.luvo_controller import LuvoController
from networks.pairwise_kldiv.kldiv_controller import KLDivController
from networks.pairwise_lstm.lstm_controller import LSTMController


class Controller(NetworkController):
    def __init__(self, 
                 setup, network, train, test,
                 clear, debug, plot, best,
                 val_data, dev_val_data, val_data_size, dev_mode):

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
        self.val_data = val_data
        self.dev_val_data = dev_val_data
        self.val_data_size = val_data_size
        self.dev_mode = dev_mode


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
                net.dev_val_data = self.dev_val_data
                net.dev_mode = self.dev_mode
                net.val_data_size = self.val_data_size

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
        plot_files(self.network, self.get_result_files())

    def get_result_files(self):
        if self.network == "all":
            regex = '*best*.pickle'
        elif self.best:
            regex = self.network + '*best*.pickle'
        else:
            # TODO: Funktioniert aktuell nicht ohne "-best" Flag
            #regex = self.network + ".pickle"
            regex = self.network + '*best*.pickle'

        files = list_all_files(get_results(), regex)

        for index, file in enumerate(files):
            files[index] = get_results(file)
        return files

if __name__ == '__main__':

    config = load_config(None, join(get_common(), 'config.cfg'))

    controller = Controller(config.getboolean('common', 'setup'), config.get('common', 'network'),
                            config.getboolean('common', 'train'), config.getboolean('common', 'test'), config.getboolean('common', 'clear'),
                            config.getboolean('common', 'debug'), config.getboolean('common', 'plot'), config.getboolean('common', 'best'),
                            config.get('validation', 'test_pickle'), config.get('validation', 'dev_pickle'),
                            config.getint('validation', 'dev_total_speakers'), config.getboolean('validation', 'dev_mode'))
    controller.run()
