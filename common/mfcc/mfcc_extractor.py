"""
The MFCCExtractor crawls the base folder and all its sub folder for wav files that
correspond with the valid speakers and extracts the MFCC's of those files.

Based on the SpectogramExtractor by Gerber, Lukic and Vogt.
"""
import os

import common.mfcc.mfcc_converter as mfcc_converter


class MfccExtractor:

    def extract_speaker_data(self, X, y, speaker_files):
        """
        Extract spectrogram and speaker names from given folder.

        :param X: return Array that saves the mel spectrogram's
        :param y: return Array that saves the speaker numbers
        :param speaker_files: dict with speaker names as keys and audio files in an array as values
        :return: the filled X, y
        """

        global_idx = 0
        curr_speaker_num = 0
        max_speakers = len(speaker_files.keys())

        # Crawl the base and all sub folders
        for speaker in speaker_files.keys():
            curr_speaker_num += 1
            speaker_uid = curr_speaker_num

            print('Extraction progress: %d/%d' % (curr_speaker_num, max_speakers))

            # Extract files
            for full_path in speaker_files[speaker]:
                extract_mfcc(full_path, X, y, global_idx, speaker_uid)
                global_idx += 1


        return X[0:global_idx], y[0:global_idx]


def extract_mfcc(wav_path, X, y, index, curr_speaker_num):
    """
    Extracts the mel spectrogram into the X array and saves the speaker into y.

    :param wav_path: the path to the wav file
    :param X: return Array that saves the mel spectrogram
    :param y: return Array that saves the speaker numbers
    :param index: the index in X and y this is stored in
    :param curr_speaker_num: the speaker number of the current speaker
    :return: a one (1) to increase the index
    """
    Sxx = mfcc_converter.mfcc(wav_path)
    for i in range(Sxx.shape[0]):
        for j in range(Sxx.shape[1]):
            X[index, i, j] = Sxx[i, j]
    y[index] = curr_speaker_num
    return 1
