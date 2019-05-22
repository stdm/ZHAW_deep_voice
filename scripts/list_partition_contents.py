import os
import h5py
import numpy as np
# import random
# random.shuffle(speaker_dict[speaker])
# random.seed(1234)
import math
# from IPython import embed

# This script lists the amount of total files, min files and max files per speaker in .h5 files
# that match the given input config settings
# 

# CONFIG SETTINGS
# ===========================================================
#
# partition_ident = 'vox2_speakers_120_test_base'
# partition_ident = 'vox2_speakers_10_test'
# partition_ident = 'vox2_speakers_5994_dev_cluster'
partition_ident = 'vox2_speakers_5994_dev_600_base'
# pickle_folder = "/mnt/all1/voxceleb2/speaker_pickles/vox2_test_120_4part"
# pickle_folder = "/mnt/all1/voxceleb2/speaker_pickles/vox2_test_10_4part"
pickle_folder = "/mnt/all1/voxceleb2/speaker_pickles/vox2_dev_5994_150part"
# pickle_folder = "/mnt/all1/voxceleb2/speaker_pickles/vox2_dev_600_15part"
# base_folder = "/home/claude/ba19_zhaw_deep_voice/archive-RNN-ANNPR-paper-Gerber/ZHAW_deep_voice"
# pickle_folder = base_folder + '/common/data/training/speaker_pickles'


# Evaluate which .h5 files need to be reprocessed
# ===========================================================
#
files = []
for root, dirs, filenames in os.walk(pickle_folder):
    for f in filenames:
        if partition_ident in f:
            files.append(f)

files.sort()

# Process .h5 files individually and generate dictionary
# containing all files of a speaker
# ===========================================================
#
for file_name in files:
    speaker_dict = dict()
    with h5py.File(pickle_folder + '/' + file_name, 'r+') as f:
        y = f['y'][:]
        speaker_names = f['speaker_names'][:]

        for i in range(len(y)):
            try:
                speaker_dict[y[i]] += 1
            except Exception as e:
                speaker_dict[y[i]] = 1

        print("Processing H5 {}...\tLen {}\tMin {}\tMax {}".format(file_name, len(set(y)), max(speaker_dict.values()), min(speaker_dict.values())))
        f.close()
