import sidekit
import os
import numpy as np

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils.load_config import *
from common.utils.paths import *
from common.utils.logger import *


class IVECController(NetworkController):
    def __init__(self):
        super().__init__("i_vector")
        self.network_file = self.name + "_100"
        self.config = load_config(None, join(get_common(), 'config.cfg'))
        self.logger = get_logger('lstm', logging.INFO)

    def train_network(self):
        self.logger.info('Runnin ivector system: training')

        ubm_list, train_idmap = self.load_data(self.config.get('train', 'pickle'))

        ubm, fs = self.train_ubm(get_training('i_vector'), self.config.get('train', 'pickle'), ubm_list, self.config.getint('ivector', 'distrib_nb'))

        self.train_total_variability(ubm, fs, self.config.getint('ivector', 'distrib_nb'), self.config.getint('ivector', 'rank_TV'), self.config.getint('ivector', 'tv_iteration'), train_idmap)

    def get_embeddings(self):
        '''
        finally, testing:
        '''
        speaker_list=self.get_validation_data_name()
        distrib_nb=self.config.getint('ivector', 'distrib_nb')
        nbThread = self.config.get('ivector', 'nbThread')
        vector_size=self.config.get('ivector', 'vector_size')
        feature_extension = 'h5'

        set_of_embeddings = []
        set_of_speakers = []
        set_of_num_embeddings = []
        checkpoints=["/TV_{}".format(self.network_file)]

        #load data:
        ubm = sidekit.Mixture()
        ubm.read(get_experiment_nets()+'/ubm_{}.h5'.format(self.network_file))
        ubm_list, test_list_long = self.load_data(self.get_validation_train_data())
        ubm_list, test_list_short = self.load_data(self.get_validation_test_data())
        tv, tv_mean, tv_sigma = sidekit.sidekit_io.read_tv_hdf5(get_experiment_nets()+"/TV_{}".format(self.network_file))

        fs = sidekit.FeaturesServer(feature_filename_structure=(
                "{dir}/{speaker_list}/feat/{{}}.{ext}".format(dir=get_training('i_vector'), speaker_list=speaker_list,
                                                              ext=feature_extension)),
            dataset_list=["energy", "cep", "vad"],
            mask="[0-12]",
            feat_norm="cmvn",
            keep_all_features=True,
            delta=True,
            double_delta=True,
            rasta=True,
            context=None)


        #exract ivectors
        test_stat_long = sidekit.StatServer(test_list_long, ubm=ubm, distrib_nb=distrib_nb, feature_size=0, index=None)
        test_stat_long.accumulate_stat(ubm=ubm, feature_server=fs, seg_indices=range(test_stat_long.segset.shape[0]),
                                  num_thread=nbThread)

        test_stat_short = sidekit.StatServer(test_list_short, ubm=ubm, distrib_nb=distrib_nb, feature_size=0, index=None)
        test_stat_short.accumulate_stat(ubm=ubm, feature_server=fs, seg_indices=range(test_stat_short.segset.shape[0]),
                                       num_thread=nbThread)

        test_iv_long = test_stat_long.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=100, num_thread=nbThread)[0]
        test_iv_short = test_stat_long.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=100, num_thread=nbThread)[0]

        #generate embeddings
        embeddings, speakers, num_embeddings=generate_embeddings(test_iv_long, test_iv_short, test_list_long,
                                                                       test_list_short, vector_size)

        set_of_embeddings.append(embeddings)
        set_of_speakers.append(speakers)
        set_of_num_embeddings.append(num_embeddings)

        return checkpoints, set_of_embeddings, set_of_speakers, set_of_num_embeddings

    def train_ubm(self, feature_dir, speaker_list, ubm_list, distrib_nb, feature_extension='h5', num_threads=10):
        '''
        training the GMM with EM-Algorithm
        '''

        self.logger.info('training UBM')

        fs = sidekit.FeaturesServer(
            feature_filename_structure=(
                "{dir}/{speaker_list}/feat/{{}}.{ext}".format(dir=feature_dir, speaker_list=speaker_list,
                                                              ext=feature_extension)),
            dataset_list=["energy", "cep", "vad"],
            mask="[0-12]",
            feat_norm="cmvn",
            keep_all_features=True,
            delta=True,
            double_delta=True,
            rasta=True,
            context=None)

        ubm = sidekit.Mixture()
        llk = ubm.EM_split(fs, ubm_list, distrib_nb, num_thread=num_threads, save_partial='gmm/ubm')
        ubm.write(get_experiment_nets()+'/ubm_{}.h5'.format(self.network_file))

        return ubm, fs

    def train_total_variability(self, ubm, fs, distrib_nb, rank_TV, tv_iteration, train_idmap, num_threads=10):

        self.logger.info('train total variability ')

        train_stat = sidekit.StatServer(train_idmap, ubm=ubm, distrib_nb=distrib_nb, feature_size=0, index=None)
        train_stat.accumulate_stat(ubm=ubm, feature_server=fs, seg_indices=range(train_stat.segset.shape[0]),
                                  num_thread=num_threads)

        tv_mean, tv, _, __, tv_sigma = train_stat.factor_analysis(rank_f=rank_TV,
                                                                 rank_g=0,
                                                                 rank_h=None,
                                                                 re_estimate_residual=False,
                                                                 it_nb=(tv_iteration, 0, 0),
                                                                 min_div=True,
                                                                 ubm=ubm,
                                                                 batch_size=100,
                                                                 num_thread=num_threads,
                                                                 save_partial="data/TV_{}".format(distrib_nb))

        sidekit.sidekit_io.write_tv_hdf5((tv, tv_mean, tv_sigma), get_experiment_nets()+"/TV_{}".format(self.network_file))

    def load_data(self, speaker_list):

        self.logger.info('load data')

        with open(join(get_training('i_vector', speaker_list), speaker_list +"_files.txt"), "r") as fh:
            ubm_list = np.array([line.rstrip() for line in fh])

        with open(join(get_training('i_vector', speaker_list), speaker_list +"_ids.txt"), "r") as fh:
            id_list = np.array([line.rstrip() for line in fh])



        tv_idmap = sidekit.IdMap()
        tv_idmap.leftids = id_list
        tv_idmap.rightids = ubm_list
        tv_idmap.start = np.empty((len(ubm_list)), dtype="|O")
        tv_idmap.stop = np.empty((len(ubm_list)), dtype="|O")



        return ubm_list, tv_idmap

