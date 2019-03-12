"""
The SpectrogramExtractor crawls the base folder and all its sub folder for wav files that
correspond with the valid speakers and extracts the spectrogram's of those files.

Based on previous work of Gerber, Lukic and Vogt.
"""
import os

import common.spectogram.spectrogram_converter as spectrogram_converter


class SpectrogramExtractor:
    def __init__(self, max_speakers, base_folder, audio_file_ending_filter, valid_speakers):
        self.max_speakers = max_speakers
        self.base_folder = base_folder
        self.audio_file_ending_filter = audio_file_ending_filter
        self.valid_speakers = valid_speakers

    def extract_speaker_data(self, X, y):

        """
        Extract spectrogram and speaker names from given folder.

        :param X: return Array that saves the mel spectrogram's
        :param y: return Array that saves the speaker numbers
        :return: the filled X, y and the speaker names
        """

        speaker_names = []
        global_idx = 0
        curr_speaker_num = -1
        old_speaker = ''
        speaker = ''

        # Crawl the base and all sub folders
        for root, directories, filenames in os.walk(self.base_folder):

            # Ignore non speaker folders
            _, root_name = os.path.split(root)
            if root_name in self.valid_speakers:
                speaker = root_name
            elif speaker == '':
                continue
        
            if speaker not in root.split(os.sep):
                continue
            
            # Check files
            for filename in filenames:
                    
                # Can't read the other wav files
                if self.audio_file_ending_filter not in filename:
                    continue

                # Extract speaker
                if speaker != old_speaker:
                    curr_speaker_num += 1
                    old_speaker = speaker
                    speaker_names.append(speaker)
                    print('Extraction progress: %d/%d' % (curr_speaker_num + 1, self.max_speakers))

                if curr_speaker_num < self.max_speakers:
                    full_path = os.path.join(root, filename)
                    # print(speaker, full_path.split(os.sep)[-3:])

                    extract_mel_spectrogram(full_path, X, y, global_idx, curr_speaker_num)
                    global_idx += 1

        return X[0:global_idx], y[0:global_idx], speaker_names


def extract_mel_spectrogram(wav_path, X, y, index, curr_speaker_num):
    """
    Extracts the mel spectrogram into the X array and saves the speaker into y.

    :param wav_path: the path to the wav file
    :param X: return Array that saves the mel spectrogram
    :param y: return Array that saves the speaker numbers
    :param index: the index in X and y this is stored in
    :param curr_speaker_num: the speaker number of the current speaker
    :return: a one (1) to increase the index
    """
    Sxx = spectrogram_converter.mel_spectrogram(wav_path)
    for i in range(Sxx.shape[0]):
        for j in range(Sxx.shape[1]):
            X[index, 0, i, j] = Sxx[i, j]
    y[index] = curr_speaker_num


# Extracts the spectrogram and discards all padded data
def extract_spectrogram(spectrogram, segment_size, frequency_elements):
    zeros = 0

    for x in spectrogram[0]:
        if x == 0.0:
            zeros += 1
        else:
            zeros = 0

    while spectrogram.shape[1] - zeros < segment_size:
        zeros -= 1

    return spectrogram[0:frequency_elements, 0:spectrogram.shape[1] - zeros]
