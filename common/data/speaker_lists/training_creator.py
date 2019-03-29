

from common.utils.paths import *
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Controller suite for Speaker clustering')
    parser.add_argument('-f', dest='filename',
                        help='Provide filename for the generated list')

    args = parser.parse_args()
    filename = args.filename

    if filename is None:
        print('please provide a filename by using the -f argument')

    else:
        forbidden_lists = {'speakers_80_stratified_test', 'speakers_80_stratified_evaluation'}
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

        for line in all_speakers:
            file.write(line)
            file.write('\n')

        file.close()

        print('Sucessfully generated list ', filename, '.txt')