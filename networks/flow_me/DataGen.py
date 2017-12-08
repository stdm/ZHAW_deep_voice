"""
    The class that supplies the network with mini-batches.
    Work of Gygax and Egly.
"""
import pickle
from random import randint

from tensorflow.examples.tutorials.mnist import input_data

from common.spectogram.spectrogram_extractor import extract_spectrogram
from common.utils.debug import *
from common.utils.paths import *


class DataGen:
    def __init__(self, config, data_set='train', pickle_no='pickle'):
        """Generator for data samples. Depending on the given config file, the DataGen instance load the TIMIT or MNIST
        dataset.

        :data_set: Defines whether training, test or validation set is loaded of the defined data set.
        """
        self.config = config
        self.batch_size = config.getint('net', 'batch_size')

        if self.config.get('exp', 'dataset') == 'timit':
            self.total_speakers = config.getint(data_set, 'total_speakers')
            self.sentences_per_speaker = config.getint(data_set, 'sentences_per_speaker')
            self.frequency_elements = config.getint('spectrogram', 'frequency_elements')
            self.duration = config.getint('spectrogram', 'duration')

            with open(get_speaker_pickle(self.config.get(data_set, pickle_no)), 'rb') as f:
                self.X, self.y, self.speaker_names = pickle.load(f)

        elif self.config.get('exp', 'dataset') == 'mnist':
            self.mnist = self.get_mnist_data_set_object(data_set)

    def create_batch(self):
        """Create batch of samples (batch size) with the mapping of labels and one representant of each class.
        """
        Zb = self.get_samples_per_class()
        Xb, yb = self.get_random_samples(self.batch_size)
        return Xb, Zb, yb

    def get_random_samples(self, random_samples=32):
        """Return random samples.
        """
        if self.config.get('exp', 'dataset') == 'timit':
            return self.get_random_timit_samples(random_samples)
        elif self.config.get('exp', 'dataset') == 'mnist':
            return self.get_random_mnist_samples(random_samples)

    def get_samples_per_class(self):
        """Returns one sample of each class.
        """
        if self.config.get('exp', 'dataset') == 'timit':
            return self.get_timit_samples_per_class()
        elif self.config.get('exp', 'dataset') == 'mnist':
            return self.get_mnist_samples_per_class()

    def get_labels(self):
        """Returns the labels of each class in the data set.
        """
        if self.config.get('exp', 'dataset') == 'timit':
            return self.speaker_names
        elif self.config.get('exp', 'dataset') == 'mnist':
            return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def get_random_timit_samples(self, random_samples=32):
        """Get the specified amount of random samples (Xb) with corresponding speaker-id (yb).
        """
        Xb = np.zeros((random_samples, 1, self.frequency_elements, self.duration), dtype=np.float32)
        yb = np.zeros(random_samples, dtype=np.int32)
        for i in range(random_samples):
            select_sentence = randint(0, self.sentences_per_speaker * self.total_speakers - 1)
            spectrogram = self.extract_spectrogram(self.X[select_sentence, 0])
            Xb[i, 0] = self.get_sample_from_sentence(spectrogram)
            yb[i] = self.y[select_sentence]
        return Xb, yb

    def get_timit_samples_per_class(self):
        """ Get one sample per class (Zb).
        """
        Zb = np.zeros((self.total_speakers, 1, self.frequency_elements, self.duration), dtype=np.float32)
        for i in range(self.total_speakers):
            select_sentence = randint(0, self.sentences_per_speaker - 1)
            spectrogram = self.extract_spectrogram(self.X[i * self.sentences_per_speaker + select_sentence, 0])
            Zb[i, 0] = self.get_sample_from_sentence(spectrogram)
        return Zb

    def get_random_mnist_samples(self, random_samples=32):
        """Get the specified amount of random samples (Xb) with corresponding label (yb).
        """
        Xb, yb = self.mnist.next_batch(random_samples)
        Xb = Xb.reshape((random_samples, 28, 28))
        Xb = np.expand_dims(Xb, axis=1)
        yb = [entry.argmax() for entry in yb]
        return Xb, yb

    def get_mnist_samples_per_class(self):
        """Get one sample per class (Zb).
        This method is a workaround, due to simplicity we used the method TensorFlow has already integrated.
        Because there is no method to retrieve a sample of a specific class, we create batches of size 1, until we retrieve
        the desired sample.
        """
        Zb = np.zeros((10, 1, 28, 28))
        fig = 0
        while fig < 10:
            z_image, z_label = self.mnist.next_batch(1)
            if z_label[0].argmax() == fig:
                Zb[z_label[0].argmax()][0] = z_image[0].reshape((28, 28))
                fig += 1
        return Zb

    def get_timit_test_set(self, sentence_pickle):
        """Returns all samples (without overlapping) of dataset .
        """
        Xb = []
        yb = []

        for sentence in range(sentence_pickle * self.total_speakers - 1):
            spectrogram = self.extract_spectrogram(self.X[sentence, 0])

            for sample in range(int(spectrogram.shape[1] / self.duration)):
                Xb.append(spectrogram[:, sample * self.duration:sample * self.duration + self.duration])
                yb.append(self.y[sentence])

        Xb = np.expand_dims(np.array(Xb), axis=1)
        yb = np.array(yb)
        return Xb, yb

    def get_sample_from_sentence(self, spectrogram):
        """ Get a random sample from within the delivered spectrogram.
        """
        sample_starting_point = randint(0, spectrogram.shape[1] - self.duration)
        return spectrogram[:, sample_starting_point:sample_starting_point + self.duration]

    def extract_spectrogram(self, spectrogram):
        return extract_spectrogram(spectrogram, self.duration, self.frequency_elements)

    def get_mnist_data_set_object(self, data_set):
        if (data_set == 'train'):
            return input_data.read_data_sets(self.config.get('data', 'mnist_path'), one_hot=True).train
        elif (data_set == 'validation'):
            return input_data.read_data_sets(self.config.get('data', 'mnist_path'), one_hot=True).validation
        elif (data_set == 'test'):
            return input_data.read_data_sets(self.config.get('data', 'mnist_path'), one_hot=True).test
