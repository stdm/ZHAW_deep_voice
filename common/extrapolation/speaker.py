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
                 frequency_elements=128, max_audio_length=800, max_files_per_partition=-1,
                 stop_after_speaker_has_no_more_files=True):
        """
        Represents a fully defined speaker in the Speaker clustering suite.

        :param frequency_elements: How many frequency elements should be in a spectrogram
        :param split_train_test: Whether or not to split the test and training data
        :param max_speakers: The maximum amount of speakers to fetch from the dataset
        :param speaker_list: The speaker list used to generate the pickle
        :param output_train: File name in which the train output pickle gets stored in
        :param output_test: File name in which the test output pickle gets stored in
        :param max_audio_length: How long the audio of the speaker can maximally be
        :param max_files_per_partition: How many files at max should be used in a partition, 
            must be greater than number of speakers, as it is divided by it, you can set -1 to load all files into initial partition
            if split_train_test is set to True, this parameter will be set to -1 per design
        :param stop_after_speaker_has_no_more_files: stops extracting after all files for a single speaker
            have been processed, even if other speakers had more files, to prevent bias.
        """
        self.frequency_elements = frequency_elements
        self.split_train_test = split_train_test
        self.max_speakers = max_speakers
        self.speaker_list = speaker_list
        self.max_audio_length = max_audio_length
        self.max_files_per_partition = max_files_per_partition
        self.stop_after_speaker_has_no_more_files = stop_after_speaker_has_no_more_files
        self.dataset = dataset

        # :split_train_test is not allowed together with max_files_per_partition
        if self.split_train_test:
            self.max_files_per_partition = -1

        # :split_train_test is not allowed with dataset=voxceleb2
        if self.split_train_test and self.dataset == "voxceleb2":
            raise Exception("Invalid arguments, split-train-test is not allowed with voxceleb2 data")

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
            self.extract(self.extract_timit_callback, get_training("TIMIT"), '_RIFF.WAV')
        elif self.dataset == "voxceleb2":
            self.extract(self.extract_voxceleb2_callback, get_training("VOXCELEB2"), '.wav')
        else:
            raise ValueError("self.dataset can only be one of ('timit', 'voxceleb2'), was " + self.dataset + ".")

        print("Done Extracting {}".format(self.speaker_list))

    def extract(self, save_callback, base_folder, file_ending):
        """
        Extracts the training and testing data from the speaker list of the dataset
        """

        # Split Train and Test Sets not available for VoxCeleb2, as the initial
        # training set is given seperate and these partitions are used to dynamically
        # generate/add new samples

        valid_speakers = self.get_valid_speakers()
        speaker_files = self.get_speaker_list_of_files(base_folder, file_ending, valid_speakers)
        speaker_count = len(valid_speakers)
        speaker_files_count = self.flattened_sum(speaker_files)
        speaker_files_per_partition = int(self.max_files_per_partition / speaker_count)

        has_files_left = speaker_files_count > 0
        partition_number = -1

        print("Extracting {} total speakers with {} files".format(speaker_count, speaker_files_count))
        
        while has_files_left:
            partition_speaker_files = dict()

            if self.max_files_per_partition == -1:
                partition_speaker_files = speaker_files
                speaker_files = dict()
            else:
                for speaker in speaker_files.keys():
                    sf = speaker_files[speaker]
                    sf_len = len(sf)
                    random.shuffle(sf)

                    if (sf_len > speaker_files_per_partition):
                        partition_speaker_files[speaker] = sf[:speaker_files_per_partition]
                        speaker_files[speaker] = sf[speaker_files_per_partition:]
                    elif sf_len == speaker_files_per_partition:
                        partition_speaker_files[speaker] = sf
                        speaker_files[speaker] = []
                    elif self.stop_after_speaker_has_no_more_files:
                        has_files_left = False
                        break
                    
                if not has_files_left:
                    return

            # extract and process spectrograms
            x, y = self.build_array_and_extract_speaker_data(partition_speaker_files)

            save_callback(x, y, valid_speakers, partition_number)

            partition_number += 1
            has_files_left = self.flattened_sum(speaker_files)

    def extract_timit_callback(self, x, y, valid_speakers, partition_number):

        # Safe Test-Data to disk
        if self.split_train_test:
            speaker_train_split = SpeakerTrainSplit(0.2)
            X_train, X_test, y_train, y_test = speaker_train_split(x, y)

            with open(get_speaker_pickle(self.output_name + '_train'), 'wb') as f:
                pickle.dump((X_train, y_train, valid_speakers), f, -1)

            with open(get_speaker_pickle(self.output_name + '_test'), 'wb') as f:
                pickle.dump((X_test, y_test, valid_speakers), f, -1)
        else:
            if partition_number == -1:
                suffix = "_cluster"
            else:
                suffix = "_cluster_" + str(partition_number)

            with open(get_speaker_pickle(self.output_name + suffix), 'wb') as f:
                pickle.dump((x, y, valid_speakers), f, -1)
        

    def extract_voxceleb2_callback(self, x, y, valid_speakers, partition_number):
            if partition_number == -1:
                suffix = "_cluster"
            else:
                suffix = "_cluster_" + str(partition_number)
                
            # store dataset
            with h5py.File(get_speaker_pickle(self.output_name + suffix, format=self.partition_format), 'w') as f:
                f.create_dataset('X', data=x)
                f.create_dataset('y', data=y)
                ds = f.create_dataset('speaker_names', (len(valid_speakers),), dtype=h5py.special_dtype(vlen=str))
                ds[:] = valid_speakers
                f.close()

            print("Done Extracting Voxceleb2 partition {}".format(partition_number))


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

    def build_array_and_extract_speaker_data(self, speaker_files):
        """
        Initialises an array based on the dimensions / count of given speaker files, frequency_elements and max_audio_length
        Extracts the spectrograms into the new array
        :param speaker_files result of dict with array elements of get_speaker_list_of_files()
        :return:
        x: the filled training data in the 4D array [Speaker, Channel, Frequency, Time]
        y: the filled testing data in a list of speaker_numbers
        """
        audio_file_count = self.flattened_sum(speaker_files)
        
        x = np.zeros((audio_file_count, 1, self.frequency_elements, self.max_audio_length), dtype=np.float32)
        y = np.zeros(audio_file_count, dtype=np.int32)
        print('build_array_and_extract_speaker_data', x.shape)

        # Extract the spectrogram's, speaker numbers and speaker names
        return SpectrogramExtractor().extract_speaker_data(x, y, speaker_files)

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

    def flattened_sum(self, dic):
        return sum(len(n) for n in dic.values())