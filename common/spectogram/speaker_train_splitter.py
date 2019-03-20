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

    # lehmacl1@2019-03-20:
    # Assumption for transition from TIMIT-coupled framework towards
    # usage of Voxceleb2 data: 1 sentence = 1 audio file
    #
    # In this method, data is being split into training and validation set.
    # Due to the nature that data per speaker does not need to be uniform,
    # it can be that we need to round the eval_size according to the number
    # of files per speaker.
    # 
    # (for eval_size = 0.2)
    # We cannot assume that exactly 20% of the total number of files will end up
    # in the validation set while the training set contains exactly 80%. Instead,
    # for each speaker we round so that the ratio per speaker in the validation 
    # and train set is as close to the eval_size as possible.
    #
    def __call__(self, X, y, speaker_files):
        valid_size = int(len(y) * self.eval_size) # 0.2y - len(y) is amount of total audio files
        train_size = int(len(y) - valid_size)     # 0.8y - len(y) is amount of total audio files

        X_new = np.zeros(X.shape, dtype=np.float32)
        y_new = np.zeros(y.shape, dtype=np.float32)

        print(X.shape, X_new.shape)
        print(y.shape, y_new.shape)

        # To avoid unnecessary resizes for the X_train/X_valid and y_train/y_valid Arrays,
        # we instead reorder X and y by filling them into X_new and y_new with the same shapes
        # but start filling the train part from index 0 and the valid part with reverse index from
        # the end (len(y) - 1)
        #
        train_index = 0
        valid_index = len(y) - 1
        total_index = 0

        for speaker in speaker_files.keys():
            speaker_files_count = len(speaker_files[speaker])
            speaker_valid_size = int(round(speaker_files_count * self.eval_size, 0))
            speaker_train_size = int(speaker_files_count - speaker_valid_size)
            # print("speaker: {}/{} ({})".format(speaker_train_size, speaker_valid_size, speaker_files_count))

            for i in range(speaker_files_count):
                # print("indices during aggregation - train: {} valid: {} total: {}".format(train_index, valid_index, total_index))

                if i > speaker_train_size:
                    X_new[valid_index] = X[total_index]
                    y_new[valid_index] = y[total_index]
                    valid_index -= 1
                else:
                    X_new[train_index] = X[total_index]
                    y_new[train_index] = y[total_index]
                    train_index += 1
                
                total_index += 1
                
        # The indices were incremented / decremented after the last step, with this correction
        # we get the last used indices for both
        #
        valid_index += 1
        train_index -= 1
        # print("indices train: {} valid: {} total: {}".format(train_index, valid_index, total_index))
            
        # Split the new sorted array into the train and validation parts
        #
        [X_train, X_valid] = np.split(X_new, [train_index]) # pylint: disable=unbalanced-tuple-unpacking
        # print("X_train ", X_train.shape)
        # print("X_valid ", X_valid.shape)

        [y_train, y_valid] = np.split(y_new, [train_index]) # pylint: disable=unbalanced-tuple-unpacking        
        # print("y_train ", y_train.shape)
        # print("y_valid ", y_valid.shape)

        return X_train, X_valid, y_train, y_valid
