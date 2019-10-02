
import os
from common.network_controller import NetworkController
from sklearn import mixture
from common.utils.pickler import load, save
from common.utils.paths import get_experiment_nets, get_speaker_pickle, get_training
from common.clustering.generate_embeddings import generate_embeddings
from common.utils.ShortUtteranceConverter import create_data_lists
import numpy as np
import h5py


class GMMController(NetworkController):
    def __init__(self, config, dev):
        super().__init__("gmm", config, dev)
        self.network_file = self.name + "_100"

    def train_network(self):
        mixture_count = self.config.getint('gmm', 'mixturecount')
        X, y = self.load_features(self.config.get('train', 'pickle'), self.config.get('train', 'pickle'), '.h5')
        model = []

        for i in range(len(X)):
            features = X[i]
            gmm = mixture.GaussianMixture(n_components=mixture_count, covariance_type='diag', n_init=1)
            gmm.fit(features)
            speaker = {'mfccs': features, 'gmm': gmm}
            model.append(speaker)

        save(model, get_experiment_nets(self.name)+'.pickle')

    def get_embeddings(self):
        X_train, y_train  = self.load_features(self.get_validation_data_name(), self.get_validation_data_name()+'_train', '.h5')
        X_test, y_test  = self.load_features(self.get_validation_data_name(), self.get_validation_data_name()+'_test', '.h5')

        model = load(get_experiment_nets(self.name)+'.pickle')

        set_of_embeddings = []
        set_of_speakers = []
        set_of_num_embeddings = []

        train_outputs = self.generate_outputs(X_train, model)
        test_outputs = self.generate_outputs(X_test, model)

        set_of_times = [np.zeros((len(y_test) + len(y_train)), dtype=int)]

        outputs, y_list, s_list = create_data_lists(False, train_outputs,test_outputs,y_train,y_test)

        embeddings, speakers, number_embeddings = generate_embeddings(outputs, y_list, len(model))

        set_of_embeddings.append(embeddings)
        set_of_speakers.append(speakers)
        set_of_num_embeddings.append(number_embeddings)
        checkpoints = [self.network_file]

        return checkpoints, set_of_embeddings, set_of_speakers, set_of_num_embeddings, set_of_times

    def generate_outputs(self, X, model):
        embeddings = []
        for sample in X:
            score_vector = []

            for speaker in model:  # find the most similar known speaker for the given test sample of a voice
                score_per_featurevector = speaker['gmm'].score(sample)  # yields log-likelihoods per feature vector
                score_vector.append(score_per_featurevector)

            embeddings.append(score_vector)

        return embeddings

    def load_features(self, speaker_list, speaker_file, file_ending ):
        X = []
        y=[]
        files_path = os.path.join(get_training('i_vector'),speaker_list,speaker_file+"_files.txt" )
        ids_path = os.path.join(get_training('i_vector'), speaker_list, speaker_file + "_ids.txt")
        with open(files_path,'r') as ff:
            with open(ids_path,'r') as fi:
                line = ff.readline()
                while line:
                    full_path = os.path.join(get_training('i_vector', speaker_list,'feat'), line.rstrip()+file_ending)
                    feature = self.load_speaker_h5(full_path)
                    id = fi.readline()
                    line = ff.readline()
                    X.append(feature.value)
                    y.append(int(id.rstrip()))
        X = np.asarray(X)
        y = np.asanyarray(y)

        return X,y


    def load_speaker_h5(self, file_path):
        f = h5py.File(file_path, 'r')
        # pick key
        key = list(f.keys())[0]
        data = f[key + "/cep"]

        return data

