import sidekit
import os
import numpy as np

from common.network_controller import NetworkController
from common.utils.load_config import *
from common.utils.paths import *


class IVECController(NetworkController):
    def __init__(self):
        super().__init__("pairwise_lstm")
        self.network_file = self.name + "_100"
        self.config = load_config(None, join(get_common(), 'config.cfg'))

    def train_network(self):
        pass

    def get_embeddings(self):
        pass


if __name__ == '__main__':

    '''
    Parameters:
    '''
    distrib_nb = 2048  # number of Gaussian distributions for each GMM
    rank_TV = 400  # Rank of the total variability matrix
    tv_iteration = 10  # number of iterations to run
    plda_rk = 400  # rank of the PLDA eigenvoice matrix
    train_dir = '/train'  # directory where to find the features
    feature_dir = get_common('feat', 'train')
    test_feature_dir = get_common('feat', 'test')
    feature_extension = 'h5'  # Extension of the feature files
    nbThread = 10  # Number of parallel process to run

    train_folder = get_common('train')

    '''
    List of files to be processed
    Needs to be adapted for TIMIT!
    '''
    with open(join(train_folder, 'speaker_list.txt'), "r") as fh:
        ubm_list = np.array([line.rstrip() for line in fh])

    tv_idmap = sidekit.IdMap()
    tv_idmap.leftids = np.array(["model_1", "model_2", "model_3", "model_4", "model_5"])
    tv_idmap.rightids = np.array(["FAEM0", "FCJF0", "FJEN0", "MARC0", "MDAC0"])
    tv_idmap.start = np.empty((5), dtype="|O")
    tv_idmap.stop = np.empty((5), dtype="|O")

    test_idmap = sidekit.IdMap()
    test_idmap.leftids = np.array(["model_1", "model_2", "model_3", "model_4", "model_5"])
    test_idmap.rightids = np.array(["FAEM0_test", "FCJF0_test", "FJEN0_test", "MARC0_test", "MDAC0_test"])
    test_idmap.start = np.empty((5), dtype="|O")
    test_idmap.stop = np.empty((5), dtype="|O")

    tv_idmap.validate()
    test_idmap.validate()
    #plda_male_idmap = sidekit.IdMap("task/plda_male_idmap.h5")
    #enroll_idmap = sidekit.IdMap("task/core_male_sre10_trn.h5")
    #test_idmap = sidekit.IdMap("task/test_sre10_idmap.h5")

    fe = sidekit.FeaturesExtractor(audio_filename_structure="common/train/{}.wav",
                                         feature_filename_structure="common/feat/train/{}.h5",
                                         sampling_frequency=None,
                                         lower_frequency=200,
                                         higher_frequency=3800,
                                         filter_bank="log",
                                         filter_bank_size=24,
                                         window_size=0.025,
                                         shift=0.01,
                                         ceps_number=20,
                                         vad="snr",
                                         snr=40,
                                         pre_emphasis=0.97,
                                         save_param=["vad", "energy", "cep"],
                                         keep_all_features=True)


    for root, directories, filenames in os.walk(train_folder):

        for filename in filenames:
            if os.path.splitext(filename)[1] == '.wav' and os. path. isfile(get_common('feat', 'train', filename)):
                fe.save(show=os.path.splitext(filename)[0], channel=0, input_audio_filename=None, output_feature_filename=None, noise_file_name=None, snr=10, reverb_file_name=None, reverb_level=-26.0)

    '''
    load accoustic data/features
    '''
    fs = sidekit.FeaturesServer(feature_filename_structure="{dir}/{{}}.{ext}".format(dir=feature_dir, ext=feature_extension),
                                dataset_list=["energy", "cep", "vad"],
                                mask="[0-12]",
                                feat_norm="cmvn",
                                keep_all_features=False,
                                delta=True,
                                double_delta=True,
                                rasta=True,
                                context=None)

    '''
    Loading keys etc
    '''
    #test_ndx = sidekit.Ndx("task/core_core_all_sre10_ndx.h5")
    #keys = sidekit.Key('task/core_core_all_sre10_cond5_key.h5')


    '''
    training the GMM with EM-Algorithm
    '''
    ubm = sidekit.Mixture()
    llk = ubm.EM_split(fs, ubm_list, distrib_nb, num_thread=nbThread, save_partial='gmm/ubm')
    ubm.write('gmm/ubm_{}.h5'.format(distrib_nb))

    '''
    StatsServer
    '''
    back_idmap = tv_idmap
    back_stat = sidekit.StatServer(back_idmap, ubm=ubm, distrib_nb=distrib_nb, feature_size=0, index=None)
    back_stat.accumulate_stat(ubm=ubm, feature_server=fs, seg_indices=range(back_stat.segset.shape[0]),
                              num_thread=nbThread)
    back_stat.write('data/stat_back.h5')

    test_stat = sidekit.StatServer(test_idmap, ubm=ubm, distrib_nb=distrib_nb, feature_size=0, index=None)
    test_stat.accumulate_stat(ubm=ubm, feature_server=fs, seg_indices=range(test_stat.segset.shape[0]),
                              num_thread=nbThread)
    test_stat.write('data/stat_test.h5')



    '''
    training tv-matrix 
    '''
    tv_mean, tv, _, __, tv_sigma = back_stat.factor_analysis(rank_f=rank_TV,
                                                           rank_g=0,
                                                           rank_h=None,
                                                           re_estimate_residual=False,
                                                           it_nb=(tv_iteration, 0, 0),
                                                           min_div=True,
                                                           ubm=ubm,
                                                           batch_size=100,
                                                           num_thread=nbThread,
                                                           save_partial="data/TV_{}".format(distrib_nb))
    sidekit.sidekit_io.write_tv_hdf5((tv, tv_mean, tv_sigma), "data/TV_{}".format(distrib_nb))



    '''
    finally, testing:
    '''
    enroll_serv = sidekit.StatServer('data/stat_back.h5'.format(distrib_nb))
    enroll_iv = enroll_serv.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=100, num_thread=nbThread)[0]
    test_serv = sidekit.StatServer('data/stat_test.h5'.format(distrib_nb))
    test_iv = test_serv.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=100, num_thread=nbThread)[0]

    ndx = sidekit.Ndx()
    ndx.modelset = np.array(["model_1", "model_2", "model_3", "model_4", "model_5"])
    ndx.segset = np.array(["FAEM0_test", "FCJF0_test", "FJEN0_test", "MARC0_test", "MDAC0_test"])
    ndx.trialmask = np.ones((ndx.modelset.shape[0], ndx.segset.shape[0]), dtype='bool')

    ndx.validate()

    '''
    cosine score
    '''
    scores_cos = sidekit.iv_scoring.cosine_scoring(enroll_iv, test_iv, ndx, wccn = None)

    '''
    plot
    '''

    key = sidekit.Key()
    key.modelset = ndx.modelset
    key.segset = ndx.segset
    key.tar = np.zeros((ndx.modelset.shape[0], ndx.segset.shape[0]), dtype='bool')
    key.tar[0, 0] = True
    key.tar[1, 1] = True
    key.tar[2, 2] = True
    key.tar[3, 3] = True
    key.tar[4, 4] = True
    key.non = np.ones((ndx.modelset.shape[0], ndx.segset.shape[0]), dtype='bool')
    key.non[0, 0] = False
    key.non[1, 1] = False
    key.non[2, 2] = False
    key.non[3, 3] = False
    key.non[4, 4] = False

    key.validate()

    dp = sidekit.DetPlot(window_style='old', plot_title='I-Vectors SRE 2010-ext male, cond 5')
    dp.set_system_from_scores(scores_cos, key, sys_name='Cosine')
    dp.create_figure()
    dp.plot_rocch_det(0)
    dp.plot_DR30_both(idx=0)
