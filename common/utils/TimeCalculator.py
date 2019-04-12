from collections import Counter

def calc_time_per_utterance(segments_speaker_train, segments_speaker_test, seg_length):
    # Count number of segments for each utterance
    train_counter = Counter(segments_speaker_train)
    test_counter = Counter(segments_speaker_test)
    train_time = [None] * len(train_counter)
    test_time = [None] * len(test_counter)

    for i, item in enumerate(train_counter):
        train_time[i] = train_counter[item]/(100/seg_length)

    for i, item in enumerate(test_counter):
        test_time[i] = test_counter[item]/(100/seg_length)

    return train_time, test_time

if __name__ == '__main__':
    segments_speaker_train = [0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,3,3,3]
    segments_speaker_test = [0,0,1,1,2,2,2,3]
    seg_length = 100
    train_time, test_time = calc_time_per_utterance(segments_speaker_train, segments_speaker_test, seg_length)
    assert train_time == [5.0, 4.0, 7.0, 3.0]
    assert test_time == [2.0, 2.0, 3.0, 1.0]