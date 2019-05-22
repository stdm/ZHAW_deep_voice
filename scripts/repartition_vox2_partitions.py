import os
import h5py
import numpy as np
import random
random.seed(1234)
import math
import pickle
# from IPython import embed

# This script repartitions given .h5 partitions for the VoxCeleb2 dataset used before
# 22.05.2019 by Christian Lauener and Claude Lehmann in their BA thesis. If the partitions
# have been created after that date, this script is likely not necessary.
# 

# CONFIG SETTINGS
# ===========================================================
#
# partition_ident = 'vox2_speakers_120_test'
# partition_ident = 'vox2_speakers_10_test'
partition_ident = 'vox2_speakers_5994_dev'

# pickle_folder = "/mnt/all1/voxceleb2/speaker_pickles/vox2_test_120_4part"
# pickle_folder = "/mnt/all1/voxceleb2/speaker_pickles/vox2_test_10_4part"
pickle_folder = "/mnt/all1/voxceleb2/speaker_pickles/vox2_dev_5994_150part"
# pickle_folder = "/mnt/all1/voxceleb2/speaker_pickles/vox2_dev_600_15part"

# base_folder = "/home/claude/ba19_zhaw_deep_voice/archive-RNN-ANNPR-paper-Gerber/ZHAW_deep_voice"
# pickle_folder = base_folder + '/common/data/training/speaker_pickles'

#   how many partitions are excepted to be produced
#   (Due to rounding errors, this can be an upper limit and is closely matched)
total_partition_count = 20

#   Take the top :speaker_count speakers in terms of available files
speaker_count = 600

# Evaluate which .h5 files need to be reprocessed
# ===========================================================
#
files = []
for root, dirs, filenames in os.walk(pickle_folder):
    for f in filenames:
        if partition_ident in f:
            if 'base' not in f:
                files.append(f)

files.sort()

#   Process .h5 files individually and generate dictionary
#   containing all files of a speaker
#   ===========================================================
#
speaker_dict_pickle_path = pickle_folder + '/' + partition_ident + '.pickle'
if os.path.exists(speaker_dict_pickle_path):
    with open(pickle_folder + partition_ident + '.pickle', 'rb') as f:
        speaker_dict = pickle.load(f)
    print("speaker_dict loaded from {}".format(speaker_dict_pickle_path))

else:
    speaker_dict = dict()

    for file_name in files:
        print("Processing H5 {}...".format(file_name))
        with h5py.File(pickle_folder + '/' + file_name, 'r+') as f:
            X = f['X'][:,0,0,0]
            y = f['y'][:]
            # speaker_names = f['speaker_names'][:]

            for i in range(len(X)):
                speaker_ident = str(y[i])

                #   For each spectrogram in the h5-file, write down the h5-filename
                #   and at which index it is located at
                try:
                    speaker_dict[speaker_ident].append(                    {
                        'index': i,
                        'speaker': speaker_ident,
                        'file_name': pickle_folder + '/' + file_name
                    })
                except KeyError:
                    speaker_dict[speaker_ident] = [{ 'index': i, 'file_name': pickle_folder + '/' + file_name }]

            f.close()


    with open(speaker_dict_pickle_path, 'wb') as f:
        pickle.dump(speaker_dict, f, -1)

    print("Saved speaker_dict to {}".format(speaker_dict_pickle_path))

#   Evaluate which speakers in the dataset have the most amount of files
#   and find the cutoff values for amount of speakers
#   ===========================================================
#
#   List with the amount of files of each speaker, to be sorted
speaker_file_counts = []
for speaker in speaker_dict.keys():
    speaker_file_counts.append(len(speaker_dict[speaker]))

speaker_file_counts.sort()

#   Cut off value as index of sorted file amount list
cut_off_value = speaker_file_counts[len(speaker_file_counts)-1-speaker_count]

#   Mark all speaker_idents/indices to remove from speaker_dict
speakers_to_remove = []
cut_off_speakers = []
for speaker in speaker_dict.keys():
    if len(speaker_dict[speaker]) < cut_off_value:
        speakers_to_remove.append(speaker)

    if len(speaker_dict[speaker]) == cut_off_value:
        cut_off_speakers.append(speaker)

print("{} speakers marked for removal due to amount of files below cut off value {}".format(len(speakers_to_remove), cut_off_value))

#   Check if there need to be speakers with the same amount of files
#   as the cut off value need to be removed
amount_of_speakers_to_remove = len(speaker_dict.keys()) - speaker_count - len(speakers_to_remove)
if amount_of_speakers_to_remove > 0:
    print("{} speakers with the same amount of files as cut off value need to be removed still".format(amount_of_speakers_to_remove))
    random.shuffle(cut_off_speakers)
    speakers_to_remove.extend(cut_off_speakers[:amount_of_speakers_to_remove])

#   Remove marked speaker from speaker_dict
print("{} speakers marked for removal due to cut off value {}".format(len(speakers_to_remove), cut_off_value))
for speaker in speakers_to_remove:
    del speaker_dict[speaker]

print("{} remaining speakers ({} expected) in speaker_dict".format(len(speaker_dict.keys()), speaker_count))

#   In order to ensure that for every speaker there are the same
#   amount of files, evaulate how many files the speaker with
#   the least amount of files has
#   ===========================================================
#

#   initialize as negative for initialization
min_speaker_file_count = -1
max_speaker_file_count = 0

for speaker in speaker_dict.keys():
    speaker_file_count = len(speaker_dict[speaker])

    if min_speaker_file_count < 0:
        min_speaker_file_count = speaker_file_count

    if speaker_file_count < min_speaker_file_count:
        min_speaker_file_count = speaker_file_count

    if speaker_file_count > max_speaker_file_count:
        max_speaker_file_count = speaker_file_count

print("Minimum amount of files over all speakers: {}".format(min_speaker_file_count))
print("Maximum amount of files over all speakers: {}".format(max_speaker_file_count))

def min_amount_of_speaker_files_left():
    min_speaker_file_count = -1

    for speaker in speaker_dict.keys():
        speaker_file_count = len(speaker_dict[speaker])

        if min_speaker_file_count < 0:
            min_speaker_file_count = speaker_file_count

        if speaker_file_count < min_speaker_file_count:
            min_speaker_file_count = speaker_file_count

    return min_speaker_file_count

#   This method takes a certain amount of speaker files and fills a new
#   partition with that amount of files from EVERY speaker in the total
#   dataset (taken from the preprocessed speaker_dict)
#   ===========================================================
#
def extract_files_into_partition(n_files, name_suffix):
    speaker_count = len(speaker_dict.keys())
    min_speaker_file_count = min_amount_of_speaker_files_left()

    if min_speaker_file_count <= 0:
        print("NO MORE FILES TO PROCESS!")
        return

    if n_files > min_speaker_file_count * speaker_count:
        n_files = min_speaker_file_count * speaker_count

    n_files_per_speaker = n_files // speaker_count
    n_files = n_files_per_speaker * speaker_count

    Xb = np.zeros([n_files, 1, 128, 800], dtype=np.float32)
    yb = np.zeros([n_files], dtype=int)

    print("Build partition {}".format(name_suffix))
    print("\tX: {}, y: {}".format(Xb.shape, yb.shape))

    base_count = 0
    for speaker in speaker_dict.keys():
        #   Evaluate files to partition
        speaker_entries = speaker_dict[speaker][:n_files_per_speaker]
        
        #   Remove used files from speaker_dict
        speaker_dict[speaker] = speaker_dict[speaker][n_files_per_speaker:]

        for entry in speaker_entries:
            # print("\t{}".format(entry))
            
            with h5py.File(entry['file_name'], 'r') as f:
                Xb[base_count,:,:,:] = f['X'][entry['index'],:,:,:]
                yb[base_count] = f['y'][entry['index']]
                f.close()

            base_count += 1

    with h5py.File(pickle_folder + '/' + partition_ident + name_suffix + '.h5', 'w') as f:
        f.create_dataset('X', data=Xb)
        f.create_dataset('y', data=yb)
        ds = f.create_dataset('speaker_names', (len(speaker_dict.keys()),), dtype=h5py.special_dtype(vlen=str))
        ds[:] = speaker_dict.keys()
        f.close()

#   Build base dataset with :n_files_per_speaker files for each speaker
#   ===========================================================
#
n_files_per_speaker = math.ceil(min_speaker_file_count / total_partition_count)
n_files = n_files_per_speaker * len(speaker_dict.keys())

print("(Base) Partition --> {}".format(n_files))
extract_files_into_partition(n_files, '_' + str(speaker_count) + '_base')

#   Build remaining datasets
#   ===========================================================
#
print("Build remaining partitions")
files_remaining = (total_partition_count - 1) * n_files
partition_counter = 0
while files_remaining > 0:
    print("Files remaining: {} (approx. {} partitions)".format(files_remaining, files_remaining // n_files))

    files_in_partition = n_files
    if files_remaining < files_in_partition:
        files_in_partition = files_remaining

    print("Partition --> {}".format(files_in_partition))
    
    extract_files_into_partition(files_in_partition, '_' + str(speaker_count) + '_base_' + str(partition_counter))
    partition_counter += 1
    files_remaining -= files_in_partition
