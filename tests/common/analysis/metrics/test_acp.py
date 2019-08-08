import unittest

from common.analysis.metrics.acp import average_cluster_purity


class TestAcp(unittest.TestCase):
    def test_equal_clusters(self):
        y_true = [0, 1, 1, 0]
        y_pred = [1, 0, 0, 1]
        self.assertAlmostEqual(average_cluster_purity(y_true, y_pred), 1.0)

    def test_one_element_predicted_clusters(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 1, 2, 3, 4, 5]
        self.assertAlmostEqual(average_cluster_purity(y_true, y_pred), 1.0)

        y_true = [0,0]
        y_pred = [0,1]
        self.assertAlmostEqual(average_cluster_purity(y_true, y_pred), 1.0)

    def test_differing_clusters(self):
        y_true = [0, 0 ,1]
        y_pred = [1, 0, 0]
        self.assertAlmostEqual(average_cluster_purity(y_true, y_pred), 2/3)

        y_true = [0, 1, 2, 1]
        y_pred = [0, 1, 1, 2]
        self.assertAlmostEqual(average_cluster_purity(y_true, y_pred), 0.75)

        y_true = [0, 1]
        y_pred = [0, 0]
        self.assertAlmostEqual(average_cluster_purity(y_true, y_pred), 0.5)


if __name__ == '__main__':
    unittest.main()
