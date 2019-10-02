import numpy as np

def _convert_to_short_utterances(x_exp, y_exp, number_of_sentences):
    x_short_list = []
    y_short_list = []
    for i in range(0, number_of_sentences):
        x_short = []
        y_short = []
        j = i
        while j < len(x_exp):
            x_short.append(x_exp[j])
            y_short.append(y_exp[j])
            j = j + number_of_sentences
        x_short_list.append(np.array(x_short))
        y_short_list.append(np.array(y_short))
    return x_short_list, y_short_list

def create_data_lists(short_utterance, X_train, X_test, y_train, y_test, s_list_train=None, s_list_test=None):
    x_list = []
    y_list = []
    s_list = []

    if (short_utterance):
        train_sentences = 8
        test_sentences = 2
        x_short_list8, y_short_list8 = _convert_to_short_utterances(X_train, y_train, train_sentences)
        x_short_list2, y_short_list2 = _convert_to_short_utterances(X_test, y_test, test_sentences)
        x_list.extend(x_short_list8)
        x_list.extend(x_short_list2)
        y_list.extend(y_short_list8)
        y_list.extend(y_short_list2)
        if(s_list_train):
            for _ in range(train_sentences):
                s_list.append(s_list_train)
        if(s_list_test):
            for _ in range(test_sentences):
                s_list.append(s_list_test)
    else:
        x_list.append(X_train)
        x_list.append(X_test)
        y_list.append(y_train)
        y_list.append(y_test)
        if(s_list_train):
            s_list.append(s_list_train)
        if (s_list_test):
            s_list.append(s_list_test)

    return x_list, y_list, s_list