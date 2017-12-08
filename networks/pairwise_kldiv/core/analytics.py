"""
    This class packages the use of the library munkres to compute the confusion matrix.

    Work of Lukic and Vogt.
"""
import sys

import munkres
import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = np.zeros([num_classes, num_classes])

    def add_predictions(self, target, predictions):
        for j in range(len(predictions)):
            self.mat[target[j], predictions[j]] += 1

    def generate_transformed_matrix(self):
        confusion = self.mat
        confusion = confusion.T
        cost_matrix = munkres.make_cost_matrix(confusion, lambda cost: sys.long_info.sizeof_digit - cost)
        m = munkres.Munkres()
        indexes = m.compute(cost_matrix)
        new_mat = np.zeros(confusion.shape)
        for i in range(len(indexes)):
            new_mat[:, i] = confusion[:, indexes[i][1]]
        return new_mat

    def calculate_accuracies(self):
        new_mat = self.generate_transformed_matrix()
        unit_mat = np.eye(len(new_mat))
        tp_mat = unit_mat * new_mat
        tps = tp_mat.sum(axis=1)
        tots = new_mat.sum(axis=1)
        accs = np.zeros(len(tots))
        for i in range(len(tps)):
            if tots[i] != 0:
                accs[i] = tps[i] / tots[i]
        return accs
