'''

Small script to generate a new list of speakers that are not in any other of the given lists

'''

from common.utils.paths import *
import argparse
import random


class ListCreator():
    def __init__(self, test, eval, t_filename, e_filename):
        self.test = test
        self.eval = eval
        self.t_filename = t_filename
        self.e_filename = e_filename

    def run(self):
        if self.test:
            self.create_testlist(self.t_filename)

        # Test network
        if self.eval:
            self.create_evallist(self.e_filename)

    def create_testlist(self, filename):

        if filename is None:
            print('Please provide a filename for the new testlist by using -tfn')
        else:
            list_size = 80
            n_male = 0
            n_female = 0
            max_female_speakers = round(30*80/100)
            max_male_speakers = list_size - max_female_speakers

            forbidden_lists = {}
            including_lists = {'speakers_40_clustering_vs_reynolds'}
            all_speakers = []

            with open(get_speaker_list('speakers_all'), 'rb') as f:
                for line in f:
                    all_speakers.append(bytes.decode(line.rstrip()))

            for list in forbidden_lists:
                try:
                    with open(get_speaker_list(list), 'rb') as f:
                        for line in f:
                            # Added bytes.decode() because python 2.x ignored leading b' while python 3.x doesn't
                            line = bytes.decode(line.rstrip())
                            if line in all_speakers:
                                all_speakers.remove(line)
                except FileNotFoundError:
                    print('List ', list, '.txt does not exist and is therefore ignored!')

            file = open(filename + '.txt', 'w')

            for list in including_lists:
                try:
                    with open(get_speaker_list(list), 'rb') as f:
                        for line in f:
                            # Added bytes.decode() because python 2.x ignored leading b' while python 3.x doesn't
                            line = bytes.decode(line.rstrip())
                            if line in all_speakers:
                                all_speakers.remove(line)
                            file.write(line)
                            file.write('\n')

                            if line.startswith('M'):
                                n_male = n_male + 1
                            else:
                                n_female = n_female + 1

                except FileNotFoundError:
                    print('List ', list, '.txt does not exist and is therefore ignored!')

            size = len(all_speakers)

            while (n_female+n_male) < list_size:
                    index = random.randrange(size)
                    line = all_speakers[index].rstrip()
                    all_speakers[index] = all_speakers[size - 1]

                    if line.startswith('M') and n_male < max_male_speakers:
                        size = size - 1
                        n_male = n_male + 1
                        file.write(line)
                        file.write('\n')

                    elif n_female < max_female_speakers:
                        size = size - 1
                        n_female = n_female + 1
                        file.write(line)
                        file.write('\n')

            file.close()

    def create_evallist(self,filename):
        if filename is None:
            print('Please provide a filename for the new testlist by using -tfn')
        else:
            list_size = 80
            n_male = 0
            n_female = 0
            max_female_speakers = round(30 * 80 / 100)
            max_male_speakers = list_size - max_female_speakers

            forbidden_lists = {'speakers_80_stratified_test'}
            including_lists = {''}
            all_speakers = []

            try:
                with open(get_speaker_list('speakers_all'), 'rb') as f:
                    for line in f:
                        all_speakers.append(bytes.decode(line.rstrip()))

                for list in forbidden_lists:
                    with open(get_speaker_list(list), 'rb') as f:
                        for line in f:
                            # Added bytes.decode() because python 2.x ignored leading b' while python 3.x doesn't
                            line = bytes.decode(line.rstrip())
                            if line in all_speakers:
                                all_speakers.remove(line)
            except FileNotFoundError:
                print('List ', line, '.txt does not exist and is therefore ignored!')

            file = open(filename + '.txt', 'w')

            for list in including_lists:
                try:
                    with open(get_speaker_list(list), 'rb') as f:
                        for line in f:
                            # Added bytes.decode() because python 2.x ignored leading b' while python 3.x doesn't
                            line = bytes.decode(line.rstrip())
                            if line in all_speakers:
                                all_speakers.remove(line)
                            file.write(line)
                            file.write('\n')

                            if line.startswith('M'):
                                n_male = n_male + 1
                            else:
                                n_female = n_female + 1
                except FileNotFoundError:
                    print('List ', list, '.txt does not exist and is therefore ignored!')

            size = len(all_speakers)

            while (n_female + n_male) < list_size:
                index = random.randrange(size)
                line = all_speakers[index].rstrip()
                all_speakers[index] = all_speakers[size - 1]

                if line.startswith('M') and n_male < max_male_speakers:
                    size = size - 1
                    n_male = n_male + 1
                    file.write(line)
                    file.write('\n')

                elif n_female < max_female_speakers:
                    size = size - 1
                    n_female = n_female + 1
                    file.write(line)
                    file.write('\n')

            file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Controller suite for Speaker clustering')
    parser.add_argument('-test', dest='test', action='store_true',
                        help='create a new testlist')
    parser.add_argument('-eval', dest='eval', action='store_true',
                        help='create a new evaluationlist.')
    parser.add_argument('-tfn', dest='t_filename',
                        help='Filename of the new testlist')
    parser.add_argument('-efn', dest='e_filename',
                        help='Filename of the new evaluationlist')

    args = parser.parse_args()

    creator = ListCreator(args.test, args.eval, args.t_filename, args.e_filename)

    creator.run()
