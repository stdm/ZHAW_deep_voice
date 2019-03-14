"""
Class to generate a callable that splits the input data into train and validation data.

Based on previous work of Gerber, Lukic and Vogt.
"""
import numpy as np


class SpeakerTrainSplit(object):
    """
    Parameters
    ----------
    eval_size : float
        Evaluation size of training set.
    """

    def __init__(self, eval_size):
        self.eval_size = eval_size

    # Annahme für Voxceleb2 Datensatz: 1 sentence = 1 audiofile
    #
    def __call__(self, X, y, speaker_files):
        valid_size = int(len(y) * self.eval_size) # 0.2y - len(y) is amount of total audio files
        train_size = int(len(y) - valid_size)     # 0.8y - len(y) is amount of total audio files

        X_train = np.zeros((train_size, 1, X[0, 0].shape[0], X[0, 0].shape[1]), dtype=np.float32)
        X_valid = np.zeros((valid_size, 1, X[0, 0].shape[0], X[0, 0].shape[1]), dtype=np.float32)
        y_train = np.zeros(train_size, dtype=np.int32)
        y_valid = np.zeros(valid_size, dtype=np.int32)

        train_index = 0
        valid_index = 0
        total_index = 0
        
        for speaker in speaker_files.keys():
            speaker_files_count = len(speaker_files[speaker])
            speaker_valid_size = int(speaker_files_count * self.eval_size)
            speaker_train_size = int(speaker_files_count - speaker_valid_size)

            for i in range(speaker_files_count):
                if i > speaker_train_size:
                    X_valid[valid_index] = X[total_index]
                    y_valid[valid_index] = y[total_index]
                    valid_index += 1
                    total_index += 1
                else:
                    X_train[train_index] = X[total_index]
                    y_train[train_index] = y[total_index]
                    train_index += 1
                    total_index += 1

        return X_train, X_valid, y_train, y_valid
