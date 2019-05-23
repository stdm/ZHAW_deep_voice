# This script iterates over different partitions of format
# <name> <name_0> <name_1> ... and adjusts the speaker_id to reflect
# the index for an imagined full dataset with all partitions as one
# 
# WARNING!!!
# This script assumes that the speakers inside the .h5 are sorted so that
# earlier speakers have lower speaker_identifiers than later ones
# 
import os
import h5py
import numpy as np
# import random
# from IPython import embed

# random.seed(1234)

# partition_ident = 'vox2_speakers_120_test_cluster'
# pickle_folder = "/mnt/all1/voxceleb2/speaker_pickles/vox2_test_120_4part"
# partition_ident = 'vox2_speakers_10_test'
# pickle_folder = "/mnt/all1/voxceleb2/speaker_pickles/vox2_test_10_4part"
partition_ident = 'vox2_speakers_5994_dev_cluster'
# pickle_folder = "/mnt/all1/voxceleb2/speaker_pickles/vox2_dev_5994_150part"
pickle_folder = "/mnt/all1/voxceleb2/speaker_pickles/vox2_dev_600_15part"

# Evaluate which .h5 files need to be reprocessed
# ===========================================================
#
files = []
for root, dirs, filenames in os.walk(pickle_folder):
    for f in filenames:
        if partition_ident in f:
            files.append(f)

files.sort()

speaker_idx = 0
for file_name in files:
    with h5py.File(pickle_folder + '/' + file_name, 'r+') as f:
        y = f['y'][:]

        is_sorted = lambda a: np.all(a[:-1] <= a[1:])

        if is_sorted(y):
            print('Partition SORTED!')
            print("\tPartition before: {} to {}".format(min(set(y)), max(set(y))))
            current_speaker_id = y[0]

            for i in range(len(y)):
                if y[i] == current_speaker_id:
                    pass
                else:
                    speaker_idx += 1
                    current_speaker_id = y[i]
                    # print("NEW! Speaker {}".format(speaker_idx))

                y[i] = speaker_idx

        else:
            print('Partition NOT SORTED!')
            exit()

        speaker_idx += 1

        f['y'][:] = y
        print("\tPartition after:  {} to {}".format(min(set(y)), max(set(y))))
        f.close()
