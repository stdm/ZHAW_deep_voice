
from common.network_controller import NetworkController
from sklearn import mixture
from common.utils.pickler import load, save
from common.utils.paths import get_experiment_nets, get_speaker_pickle


class GMMController(NetworkController):
    def __init__(self, config):
        super().__init__("gmm", config)
        self.network_file = self.name + "_100"


    def train_network(self):
        '''build a 16 component diagonal covariance GMM from the given features (usually 13 MFCCs)'''
        mixture_count = self.config.getint('gmm', 'mixturecount')
        X, y, speaker_names = load(get_speaker_pickle(self.config.get('train', 'mfcc_pickle')))
        model = []

        for i in range(len(X)):
            features = X[i]
            gmm = mixture.GaussianMixture(n_components=mixture_count, covariance_type='diag', n_init=1)
            gmm.fit(features.transpose())
            speaker = {'mfccs': features, 'gmm': gmm}
            model.append(speaker)

        save(model, get_experiment_nets(self.name))


    def get_embeddings(self):
        X, y, speaker_names = load(self.get_validation_data_name())
        model = load(get_experiment_nets(self.name))

        for sample in X:
            for speaker in model:  # find the most similar known speaker for the given test sample of a voice
                score_per_featurevector = speaker['gmm'].score(sample)  # yields log-likelihoods per feature vector




