import unittest

from common.analysis.metrics.ari import adjusted_rand_index


class TestAri(unittest.TestCase):
    def test_equal_clusters(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        self.assertAlmostEqual(adjusted_rand_index(y_true, y_pred), 1.0)

    def test_equal_results_with_swapped_clusters(self):
        y_true = [0, 0, 1, 2]
        y_pred = [0, 0, 1, 1]
        self.assertAlmostEqual(adjusted_rand_index(y_true, y_pred), adjusted_rand_index(y_pred, y_true))

    def test_bigger_clusters(self):
        y_true = [1, 2, 3, 3, 2, 1, 1, 3, 3, 1, 2, 2]
        y_pred = [3, 2, 3, 2, 2, 1, 1, 2, 3, 1, 3, 1]
        self.assertAlmostEqual(adjusted_rand_index(y_true, y_pred), 1 / 12)

    def test_one_element_predicted_clusters(self):
        y_true = [0, 0, 0, 0]
        y_pred = [0, 1, 2, 3]
        self.assertAlmostEqual(adjusted_rand_index(y_true, y_pred), 0.0)


if __name__ == '__main__':
    unittest.main()
