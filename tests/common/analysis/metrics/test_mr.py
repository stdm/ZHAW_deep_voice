import unittest

from common.analysis.metrics.mr import misclassification_rate


class TestMr(unittest.TestCase):
    def test_equal_clusters(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 2, 2]
        self.assertAlmostEqual(misclassification_rate(y_true, y_pred), 0.0)

    def test_differing_clusters(self):
        y_true = [1, 1, 1, 1]
        y_pred = [2, 2, 1, 1]
        self.assertAlmostEqual(misclassification_rate(y_true, y_pred), 0.5)

        y_true = [1, 2, 2, 4]
        y_pred = [1, 1, 1, 1]
        self.assertAlmostEqual(misclassification_rate(y_true, y_pred), 0.5)

    def test_equal_results_with_swapped_clusters(self):
        y_true = [1, 1, 1, 1]
        y_pred = [1, 2, 3, 4]
        self.assertAlmostEqual(misclassification_rate(y_true, y_pred), misclassification_rate(y_pred, y_true))

    def test_bigger_clusters(self):
        y_true = [0, 0, 0, 1, 1, 1, 1, 2, 2]
        y_pred = [7, 7, 3, 3, 2, 2, 1, 0, 0]
        self.assertAlmostEqual(misclassification_rate(y_true, y_pred), 1 / 3)

        y_true = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1]
        y_pred = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        self.assertAlmostEqual(misclassification_rate(y_true, y_pred), 5 / 13)


if __name__ == '__main__':
    unittest.main()
