import unittest

from common.analysis.metrics.der import diarization_error_rate


class TestDer(unittest.TestCase):
    def test_differing_clusters(self):
        y_true = [1, 1, 2, 2]
        y_pred = [1, 2, 2, 2]
        times = [8, 2, 8, 2]
        self.assertAlmostEqual(diarization_error_rate(y_true, y_pred, times), 0.1)

    def test_one_element_predicted_clusters(self):
        y_true = [1, 1, 2, 2]
        y_pred = [1, 2, 3, 4]
        times = [8, 2, 8, 2]
        self.assertAlmostEqual(diarization_error_rate(y_true, y_pred, times), 0.2)

if __name__ == '__main__':
    unittest.main()
