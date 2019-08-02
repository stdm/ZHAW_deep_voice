from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Segment, Annotation

def diarization_error_rate(y_true, y_pred, times):
    '''
    :param y_true: Ground truth speakers per utterance
    :param y_pred: Predicted speakers per utterance
    :param times: time per utterance in seconds (the seconds can be a float, they will be converted and rounded to
        integer milliseconds)
    :return: The Diarization Error Rate (DER)
    '''
    metric = DiarizationErrorRate()
    reference = _generate_annotations(y_true, times)
    hypothesis = _generate_annotations(y_pred, times)
    value = metric(reference, hypothesis)
    return value

def _generate_annotations(utterances, times):
    annotation = Annotation()
    start = 0
    for u, utterance in enumerate(utterances):
        end = start + round(times[u] * 1000)
        annotation[Segment(start, end)] = utterance
        start = end
    return annotation


if __name__ == '__main__':
    def flt_eq(x, y):
        return abs(x - y) < 1e-5

    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 2, 2]
    times = [8, 2, 8, 2]
    assert flt_eq(diarization_error_rate(y_true, y_pred, times), 0.1)

    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 3, 4]
    times = [8, 2, 8, 2]
    assert flt_eq(diarization_error_rate(y_true, y_pred, times), 0.2)