from collections import Counter

ONE_SECOND = 100

def calc_time_long_short_utterances(segments_train, segments_test, seg_length):
    '''
    :param segments_traing: A list containing the utterances corresponding to the segments.
        This list contains the speakers for long utterances (8 sentences)
    :param segments_test: A list containing the utterances corresponding to the segments.
        This list contains the speakers for short utterances (2 sentences)
    :param seg_length: the length of a segment
    :return: an array that contains the length of each utterance. The first half of the array contains
        the time per long utterance, the time per short utterance are in the second half.
    '''

    # Count number of segments for each utterance
    train_time = calc_time_per_utterance(segments_train, seg_length)
    test_time = calc_time_per_utterance(segments_test, seg_length)

    total_time = []
    total_time.extend(train_time)
    total_time.extend(test_time)

    return total_time

def calc_time_per_utterance(segments, seg_length):
    '''
    :param segments: A list containing the utterances corresponding to the segments
    :param seg_length: the length of a segment
    :return: an array that contains the length of each utterance
    '''
    counter = Counter(segments)
    time = [None] * len(counter)
    for i, item in enumerate(counter):
        time[i] = counter[item]*seg_length/ONE_SECOND
    return time

if __name__ == '__main__':
    segments_train = [0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,3,3,3]
    segments_test = [0,0,1,1,2,2,2,3]
    seg_length = 100
    time = calc_time_long_short_utterances(segments_train, segments_test, seg_length)
    assert time == [5.0, 4.0, 7.0, 3.0, 2.0, 2.0, 3.0, 1.0]