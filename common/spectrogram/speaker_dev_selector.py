import numpy as np
import random
from common.utils.pickler import load


def load_test_data(data_path):
    x, y, s_list = load(data_path)
    return x,y, s_list


'''
This method randomly chooses a number of speakers out of a test set.
As of now, this method should only be used for dev-tests to prevent overfitting during development. (see BA_2019 for details)
'''
def load_dev_test_data(data_path, number_speakers, sentences):
    X, y, s_list = load(data_path)

    if number_speakers < 80:
        X_sampled = np.zeros(X.shape)
        y_sampled = np.zeros(y.shape, dtype=np.int8)
        for i in range(number_speakers):
            index = random.randrange(80-i)

            X_extraced, y_extracted, X, y = get_sentences_for_speaker_index(X, y, index, i, sentences)
            X_sampled[i*sentences:i*sentences+sentences] = X_extraced
            y_sampled[i*sentences:i*sentences+sentences] = y_extracted

        X, speakers = X_sampled[:(number_speakers * sentences)], y_sampled[:(number_speakers * sentences)]

    return X, speakers, s_list


'''
In this method, all sentences from a speaker with a given id are extracted.
The sentences will be put at the place of the last element (or the number of picks away from the end) and the updated
list of speakers will be returned as well! 
'''
def get_sentences_for_speaker_index(x, y, index, pick, sentences):
    x_sampled = np.zeros(x.shape)
    y_sampled = np.zeros(y.shape)
    size = len(x)
    for j in range(0, sentences):
        x_sampled[j] = x[index*sentences+j]
        y_sampled[j] = y[index * sentences + j]
        x[index*sentences + j] = x[size-sentences-pick*sentences+j]
        y[index * sentences + j] = y[size-sentences-pick*sentences+j]

    return x_sampled[:sentences], y_sampled[:sentences], x, y

