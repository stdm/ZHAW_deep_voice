import numpy as np



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