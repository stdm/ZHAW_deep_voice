"""
A Speaker contains all needed information and methods to create the pickle file used for training.

Based on previous work of Gerber, Lukic and Vogt, adapted by Heusser
"""
import random
from math import ceil

import pickle
import h5py
import numpy as np

from common.spectogram.speaker_train_splitter import SpeakerTrainSplit
from common.spectogram.spectrogram_extractor import SpectrogramExtractor
from common.utils.paths import *

random.seed(1234)

class Speaker:
    def __init__(self, split_train_test, max_speakers, speaker_list, dataset, output_name=None,
                 frequency_elements=128, max_audio_length=800, speakers_per_partition=40):
        """
        Represents a fully defined speaker in the Speaker clustering suite.

        :param frequency_elements: How many frequency elements should be in a spectrogram
        :param split_train_test: Whether or not to split the test and training data
        :param max_speakers: The maximum amount of speakers to fetch from the dataset
        :param speaker_list: The speaker list used to generate the pickle
        :param output_train: File name in which the train output pickle gets stored in
        :param output_test: File name in which the test output pickle gets stored in
        :param max_audio_length: How long the audio of the speaker can maximally be
        """
        self.frequency_elements = frequency_elements
        self.split_train_test = split_train_test
        self.max_speakers = max_speakers
        self.speaker_list = speaker_list
        self.max_audio_length = max_audio_length
        self.speakers_per_partition = speakers_per_partition
        self.dataset = dataset

        # :partition_format for VoxCeleb2 is h5, due to the 2GB limit of .pickle files.
        # In case of different datasets, h5 can be used to more easily work with more data
        #
        if self.dataset == "voxceleb2":
            self.partition_format = '.h5'
        else:
            self.partition_format = '.pickle'

        if output_name is None:
            self.output_name = speaker_list
        else:
            self.output_name = output_name

    def safe_to_pickle(self):
        """
        Fetches all data for this speaker from the dataset and safes it inside of a pickle.
        """
        print("Extracting {}".format(self.speaker_list))

        if self.dataset == "timit":
            # Extract the spectrogram's, speaker numbers and speaker names
            X, y, speaker_names = self.extract_timit()

            # Safe Test-Data to disk
            if self.split_train_test:
                speaker_train_split = SpeakerTrainSplit(0.2)
                X_train, X_test, y_train, y_test = speaker_train_split(X, y)

                with open(get_speaker_pickle(self.output_name + '_train'), 'wb') as f:
                    pickle.dump((X_train, y_train, speaker_names), f, -1)

                with open(get_speaker_pickle(self.output_name + '_test'), 'wb') as f:
                    pickle.dump((X_test, y_test, speaker_names), f, -1)
            else:
                with open(get_speaker_pickle(self.output_name + '_cluster'), 'wb') as f:
                    pickle.dump((X, y, speaker_names), f, -1)

        elif self.dataset == "voxceleb2":
            # take list of all speakers and shuffle them
            #
            list_of_speakers = self.get_valid_speakers()
            
            # Extract initial dataset
            #
            initial_dataset_speakers = list_of_speakers[0:self.speakers_per_partition]
            print("Extracting Voxceleb2 Initial Dataset")

            # extract slice into pickle (w/ or w/o train_test split)
            #
            X, y, speaker_names = self.extract_voxceleb2(valid_speakers=initial_dataset_speakers, offset=0)

            # Split Train and Test Sets not available for VoxCeleb2, as the initial
            # training set is given seperate and these partitions are used to dynamically
            # generate/add new samples
            #
            # When data is located in a different folder:
            # with h5py.File(get_speaker_pickle('/mnt/all1/voxceleb2/speaker_pickles/' + self.output_name + '_cluster', format=self.partition_format), 'w') as f:
            with h5py.File(get_speaker_pickle(self.output_name + '_cluster', format=self.partition_format), 'w') as f:
                f.create_dataset('X', data=X)
                f.create_dataset('y', data=y)
                ds = f.create_dataset('speaker_names', (len(speaker_names),), dtype=h5py.special_dtype(vlen=str))
                ds[:] = speaker_names
                f.close()

            print("Done Extracting Voxceleb2 Initial Dataset")

            list_of_speakers = list_of_speakers[self.speakers_per_partition:]
            random.shuffle(list_of_speakers)

            # slice the speaker list into partitions N partitions, this is calculataed by the total number of 
            # speakers and divided by the expected elements per partition, rounded up
            #
            speaker_count = len(list_of_speakers)
            speaker_splits = np.array_split(list_of_speakers, ceil(speaker_count / self.speakers_per_partition))
            speaker_offset = len(speaker_names)

            for i in range(len(speaker_splits)):
                print("Extracting Voxceleb2 SpeakerSplit {}".format(i))

                # extract slice into pickle (w/ or w/o train_test split)
                #
                X, y, speaker_names = self.extract_voxceleb2(valid_speakers=speaker_splits[i], offset=speaker_offset)

                # Split Train and Test Sets not available for VoxCeleb2, as the initial
                # training set is given seperate and these partitions are used to dynamically
                # generate/add new samples
                #
                # When data is located in a different folder:
                # with h5py.File(get_speaker_pickle('/mnt/all1/voxceleb2/speaker_pickles/' + self.output_name + '_cluster_' + str(i), format=self.partition_format), 'w') as f:
                with h5py.File(get_speaker_pickle(self.output_name + '_cluster_' + str(i), format=self.partition_format), 'w') as f:
                    f.create_dataset('X', data=X)
                    f.create_dataset('y', data=y)
                    ds = f.create_dataset('speaker_names', (len(speaker_names),), dtype=h5py.special_dtype(vlen=str))
                    ds[:] = speaker_names
                    f.close()

                speaker_offset += len(speaker_names)
                print("Done Extracting Voxceleb2 SpeakerSplit {}".format(i))

        else:
            raise ValueError("self.dataset can only be one of ('timit', 'voxceleb2'), was " + self.dataset + ".")

        print("Done Extracting {}".format(self.speaker_list))
        print("Saved to pickle.\n")

    def extract_timit(self):
        """
        Extracts the training and testing data from the speaker list of the TIMIT Dataset
        :return:
        x: the filled training data in the 4D array [Speaker, Channel, Frequency, Time]
        y: the filled testing data in a list of speaker_numbers
        valid_speakers: list of all speakers in this dataset
        """
        # Add all valid speakers
        valid_speakers = self.get_valid_speakers()
        speaker_files = self.get_speaker_list_of_files(get_training("TIMIT"), '_RIFF.WAV', valid_speakers)

        # Extract the spectrogram's, speaker numbers and speaker names
        x, y = self.build_array_and_extract_speaker_data(speaker_files)
        return x, y, valid_speakers

    def extract_voxceleb2(self, valid_speakers, offset):
        """
        Extracts the training and testing data from the speaker list of the VoxCeleb2 Dataset
        :return:
        x: the filled training data in the 4D array [Speaker, Channel, Frequency, Time]
        y: the filled testing data in a list of speaker_numbers
        valid_speakers: list of all speakers in this dataset
        """

        # When data is located in a different folder:
        # speaker_files = self.get_speaker_list_of_files('/mnt/all1/voxceleb2/test/aac', '.wav', valid_speakers)
        speaker_files = self.get_speaker_list_of_files(get_training("VOXCELEB2"), '.wav', valid_speakers)
        
        # Extract the spectrogram's, speaker numbers and speaker names
        x, y = self.build_array_and_extract_speaker_data(speaker_files, offset)
        return x, y, valid_speakers

    def get_valid_speakers(self):
        """
        Return an array with all speakers
        :return:
        valid_speakers: array with all speakers
        """
        # Add all valid speakers
        valid_speakers = []
        with open(get_speaker_list(self.speaker_list), 'rb') as f:
            for line in f:
                # Added bytes.decode() because python 2.x ignored leading b' while python 3.x doesn't
                valid_speakers.append(bytes.decode(line.rstrip()))

        return valid_speakers

    def get_speaker_list_of_files(self, base_folder, file_ending, valid_speakers):
        """
        Return a two dimensional array with all speakers and all of their audio files
        :return:
        speaker_files: dictionary with speaker_name as key and an array of full file paths to all audio files of that speaker
        """
        result = {}
        speaker = ""

        for root, _, filenames in os.walk(base_folder):
            # Ignore non speaker folders
            _, root_name = os.path.split(root)
            if root_name in valid_speakers:
                speaker = root_name
                result[speaker] = []
            elif speaker == '':
                continue
        
            if speaker not in root.split(os.sep):
                continue
            
            # Check files
            for filename in filenames:
                # Can't read the other wav files
                if file_ending not in filename:
                    continue

                full_path = os.path.join(root, filename)
                result[speaker].append(full_path)
                
        return result

    def build_array_and_extract_speaker_data(self, speaker_files, offset=0):
        """
        Initialises an array based on the dimensions / count of given speaker files, frequency_elements and max_audio_length
        Extracts the spectrograms into the new array
        :param speaker_files result of dict with array elements of get_speaker_list_of_files()
        :return:
        x: the filled training data in the 4D array [Speaker, Channel, Frequency, Time]
        y: the filled testing data in a list of speaker_numbers
        """
        audio_file_count = sum(len(n) for n in speaker_files.values())
        
        x = np.zeros((audio_file_count, 1, self.frequency_elements, self.max_audio_length), dtype=np.float32)
        y = np.zeros(audio_file_count, dtype=np.int32)
        print('build_array_and_extract_speaker_data', x.shape)

        # Extract the spectrogram's, speaker numbers and speaker names
        return SpectrogramExtractor().extract_speaker_data(x, y, speaker_files, offset)

    def is_pickle_saved(self):
        """
        Whether or not a corresponding pickle file already exist.
        :return: true if it exists, false otherwise
        """
        if self.split_train_test:
            # When data is located in a different folder:
            # return path.exists(get_speaker_pickle('/mnt/all1/voxceleb2/speaker_pickles/' + self.output_name + '_train', format=self.partition_format))
            return path.exists(get_speaker_pickle(self.output_name + '_train', format=self.partition_format))
        else:
            # When data is located in a different folder:
            # return path.exists(get_speaker_pickle('/mnt/all1/voxceleb2/speaker_pickles/' + self.output_name + '_cluster', format=self.partition_format))
            return path.exists(get_speaker_pickle(self.output_name + '_cluster', format=self.partition_format))
