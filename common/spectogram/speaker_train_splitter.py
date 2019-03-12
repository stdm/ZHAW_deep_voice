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
    sentences : int
        Number of sentences in training set for each speaker.
    """

    def __init__(self, eval_size, sentences):
        self.eval_size = eval_size
        self.sentences = sentences

    # Annahme für Voxceleb2 Datensatz: 1 sentence = 1 audiofile
    #
    def __call__(self, X, y):
        valid_size = int(len(y) * self.eval_size) # 0.2y
        train_size = int(len(y) - valid_size)     # 0.8y
        X_train = np.zeros((train_size, 1, X[0, 0].shape[0], X[0, 0].shape[1]), dtype=np.float32)
        X_valid = np.zeros((valid_size, 1, X[0, 0].shape[0], X[0, 0].shape[1]), dtype=np.float32)
        y_train = np.zeros(train_size, dtype=np.int32)
        y_valid = np.zeros(valid_size, dtype=np.int32)

        train_index = 0
        valid_index = 0
        # lehmacl1@2019-03-05: Muss für Voxceleb umgeschrieben werden und dynamisch nachgerechnet werden
        # anhand der Anzahl effektiver Sentences pro Sprecher

        # TODO: Aufspalten nach Speaker in Train/Valid Set
        #
        nth_elem = self.sentences - self.sentences * self.eval_size # 8
        
        for i in range(len(y)):
            if i % self.sentences >= nth_elem:
                X_valid[valid_index] = X[i]
                y_valid[valid_index] = y[i]
                valid_index += 1
            else:
                X_train[train_index] = X[i]
                y_train[train_index] = y[i]
                train_index += 1

        return X_train, X_valid, y_train, y_valid
