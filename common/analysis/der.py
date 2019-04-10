from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Segment, Annotation

def diarization_error_rate(reference, hypothesis):
    metric = DiarizationErrorRate()
    value = metric(reference, hypothesis)
    return value


if __name__ == '__main__':
    def flt_eq(x, y):
        return abs(x - y) < 1e-5

    reference = Annotation()
    reference[Segment(0, 8)] = 1
    reference[Segment(8, 10)] = 1
    reference[Segment(10, 18)] = 2
    reference[Segment(18, 20)] = 2
    hypothesis = Annotation()
    hypothesis[Segment(0, 8)] = 1
    hypothesis[Segment(8, 10)] = 2
    hypothesis[Segment(10, 18)] = 2
    hypothesis[Segment(18, 20)] = 2
    assert flt_eq(diarization_error_rate(reference, hypothesis), 0.1)