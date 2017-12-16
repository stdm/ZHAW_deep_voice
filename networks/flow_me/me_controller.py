"""
The controller to train and test the fs17_ba_gygax_egli network
"""

import networks.flow_me.DataGen as DataGen
from common.analysis.old_plots import *
from common.network_controller import NetworkController
from common.utils.load_config import *
from common.utils.logger import *
from common.utils.paths import *
from .network_runnable import train_network
from .nn.restore_session import *


class MEController(NetworkController):
    def __init__(self, clear, debug, nchw):
        super().__init__("flow_me")
        self.clear = clear
        self.debug = debug
        self.nchw = nchw
        self.config_path = '022_100_densefactor'

    def train_network(self):
        train_network(get_configs(self.config_path), self.clear, self.debug, self.nchw)

    def get_embeddings(self):
        # Prepare return values
        checkpoint_names = []
        set_of_embeddings = []
        set_of_speakers = []
        speaker_numbers = []

        # Correct list (Hacky hack because of the config structure in this neural network)
        list_dict = {
            "speakers_40_clustering_vs_reynolds": 1,
            "speakers_60_clustering": 2,
            "speakers_80_clustering": 3
        }

        list_number = list_dict[self.val_data]
        print('List ' + str(list_number))
        for checkpoint in list(range(999, 30001, 1000)):
            checkpoint_str = str(checkpoint)
            print('Checkpoint ' + checkpoint_str)
            embeddings, map_labels = self.test(checkpoint, list_number, self.config_path)
            checkpoint_name = self.name + ': checkpoint: ' + checkpoint_str

            # Fill return values
            checkpoint_names.append(checkpoint_name)
            set_of_embeddings.append(embeddings)
            set_of_speakers.append(map_labels)
            speaker_numbers.append(len(set(map_labels)) * 2)

        return checkpoint_names, set_of_embeddings, set_of_speakers, speaker_numbers

    def test(self, checkpoint=0, list=1, config_path=None):

        if config_path is None:
            config_path = self.config_path

        path_master_config = get_configs('master')
        config = load_config(path_master_config, get_configs(config_path))
        logger = get_logger('cluster', logging.INFO)

        list = 'test_list' + str(list)

        # Log Output folder
        folder_name = str(config.get(list, 'total_speakers')) + "_speakers"
        checkpoint_folder = 'checkpoint_{:05d}'.format(checkpoint)
        output_folder = get_experiment_logs("flow_me", folder_name, checkpoint_folder)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        add_file_handler(logger, join(output_folder, 'mr.log'))

        # Load test data
        logger.info("Loading Test Data")
        data_gen_8 = DataGen.DataGen(config, data_set=list, pickle_no='pickle1')
        data_gen_2 = DataGen.DataGen(config, data_set=list, pickle_no='pickle2')

        input_data_8, map_labels_8 = data_gen_8.get_timit_test_set(
            sentence_pickle=config.getint(list, 'sentences_pickle1'))
        input_data_2, map_labels_2 = data_gen_2.get_timit_test_set(
            sentence_pickle=config.getint(list, 'sentences_pickle2'))

        labels = data_gen_8.get_labels()  # Both data_gen instances should return the same labels

        if not self.nchw:
            input_data_8 = np.transpose(input_data_8, axes=(0, 2, 3, 1))
            input_data_2 = np.transpose(input_data_2, axes=(0, 2, 3, 1))

        # Load Session
        output_layer_name = config.get('test', 'output_layer')
        output_layer_name = output_layer_name.replace('\\', '/')
        flow_me_nets = get_experiment_nets("flow_me")
        path_meta_graph = join(flow_me_nets, 'model.ckpt-' + str(checkpoint) + '.meta')
        path_ckpt = join(flow_me_nets, 'model.ckpt-' + str(checkpoint))

        sess, input_layer, output_layer, train_mode = restore_session(path_meta_graph, path_ckpt, output_layer_name)
        logger.info("Restored Session:\tCheckpoint = {:}\t{:}".format(checkpoint, folder_name))

        # Pass samples trough network, get embeddings
        embeddings_8 = sess.run(output_layer, feed_dict={input_layer: input_data_8, train_mode: False})
        embeddings_2 = sess.run(output_layer, feed_dict={input_layer: input_data_2, train_mode: False})
        sess.close()
        logger.info("Created Embeddings")

        # Reduce to mean, merge to one list of embeddings
        # Init data structures according to used embeddings dimension
        if 'l10_dense' in config.get('test', 'output_layer'):
            embeddings_mean_8 = np.zeros((config.getint(list, 'total_speakers'), int(config.getfloat(
                'net', 'dense10_factor') * config.getint('train', 'total_speakers'))))
            embeddings_mean_2 = np.zeros((config.getint(list, 'total_speakers'), int(config.getfloat(
                'net', 'dense10_factor') * config.getint('train', 'total_speakers'))))
        elif 'l11_dense' in config.get('test', 'output_layer'):
            embeddings_mean_8 = np.zeros((config.getint(list, 'total_speakers'), int(config.getfloat(
                'net', 'dense11_factor') * config.getint('train', 'total_speakers'))))
            embeddings_mean_2 = np.zeros((config.getint(list, 'total_speakers'), int(config.getfloat(
                'net', 'dense11_factor') * config.getint('train', 'total_speakers'))))
        else:
            embeddings_mean_8 = np.zeros((config.getint(list, 'total_speakers'), int(config.getfloat(
                'net', 'dense7_factor') * config.getint('train', 'total_speakers'))))
            embeddings_mean_2 = np.zeros((config.getint(list, 'total_speakers'), int(config.getfloat(
                'net', 'dense7_factor') * config.getint('train', 'total_speakers'))))

        # Sum of all embeddings
        for embedding, label in zip(embeddings_8, map_labels_8):
            embeddings_mean_8[label] += embedding
        for embedding, label in zip(embeddings_2, map_labels_2):
            embeddings_mean_2[label] += embedding

        # Divsion trough count of embeddings from the same speaker
        unique_8, counts_8 = np.unique(map_labels_8, return_counts=True)
        for label, count in zip(unique_8, counts_8):
            embeddings_mean_8[label] /= count

        unique_2, counts_2 = np.unique(map_labels_2, return_counts=True)
        for label, count in zip(unique_2, counts_2):
            embeddings_mean_2[label] /= count

        # Concatenate to one date set for analysis
        embeddings = np.concatenate((embeddings_mean_8, embeddings_mean_2))
        map_labels = np.concatenate((unique_8, unique_2))

        return embeddings, map_labels
