'''
This class picks a number of speakers randomly from a set of speakers.
The idea is to generate a set of 40 speakers randomly based on a list of 200 speakers for evaluation (which is not testing!)

created 03.2019, Jan Sonderegger
'''


import numpy as np
import random


class EvalData_Picker(object):
    """
    Parameters
    ----------

    """

    def __init__(self, eval_size, sentences):
        self.eval_size = eval_size
        self.sentences = sentences

    def __call__(self, X, y, net=None):
        X_eval = np.zeros((self.eval_size, 1, X[0, 0].shape[0], X[0, 0].shape[1]), dtype=np.float32)
        y_eval = np.zeros(self.eval_size, dtype=np.int32)

        eval_index = 0
        # nth_elem = self.sentences - self.sentences * self.eval_size
        size = len(y)
        for i in range(1,self.eval_size):
            index = random.randrange(size)
            X_eval[eval_index] = X[index]
            X[index] = X[size]
            y_eval[eval_index] = y[index]
            y[index] = y[size]
            size = size - 1
            eval_index = eval_index + 1

        return X_eval, y_eval