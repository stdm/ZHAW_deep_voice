import os
import h5py
import numpy as np
import random
# from IPython import embed

random.seed(1234)

def get_valid_speakers(path):
    valid_speakers = []
    with open(path, 'rb') as f:
        for line in f:
            # Added bytes.decode() because python 2.x ignored leading b' while python 3.x doesn't
            valid_speakers.append(bytes.decode(line.rstrip()))

    return valid_speakers

partition_ident = 'vox2_speakers_10_test'
base_folder = "/home/claude/ba19_zhaw_deep_voice/archive-RNN-ANNPR-paper-Gerber/ZHAW_deep_voice"
pickle_folder = base_folder + '/common/data/training/speaker_pickles'
speaker_list_path = base_folder + '/common/data/speaker_lists/' + partition_ident + '.txt'

num_files_per_speaker_in_base_set = 2
speaker_list = get_valid_speakers(speaker_list_path)

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

# Process .h5 files individually and generate dictionary
# containing all files of a speaker
# ===========================================================
#
speaker_dict = dict()

for file_name in files:
    print("Processing H5 {}...".format(file_name))
    with h5py.File(pickle_folder + '/' + file_name, 'r+') as f:
        X = f['X'][:,:,:,:]
        y = f['y'][:]
        speaker_names = f['speaker_names'][:]
        
        for i in range(len(X)):
            speaker_ident = str(y[i])

            try:
                speaker_dict[speaker_ident].append(                    {
                    'index': i,
                    'file_name': pickle_folder + '/' + file_name
                })
            except KeyError:
                speaker_dict[speaker_ident] = [
                    {
                        'index': i,
                        'file_name': pickle_folder + '/' + file_name
                    }
                ]

        f.close()

def extract_files_into_partition(n_files, name_suffix):
    Xb = np.zeros([n_files, 1, 128, 800], dtype=np.float32)
    yb = np.zeros([n_files], dtype=int)

    print("Build partition {}".format(name_suffix))
    print("\tX: {}, y: {}".format(Xb.shape, yb.shape))

    base_count = 0
    for speaker in speaker_dict.keys():
        # Evaluate files to partition
        random.shuffle(speaker_dict[speaker])
        speaker_entry = speaker_dict[speaker][:num_files_per_speaker_in_base_set]
        
        # Remove used files from speaker_dict
        speaker_dict[speaker] = speaker_dict[speaker][num_files_per_speaker_in_base_set:]
        
        # print("Processing Speaker {}...".format(speaker))
        for entry in speaker_entry:
            # print("\t{}".format(entry))

            with h5py.File(entry['file_name'], 'r+') as f:
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

# Build base dataset with :num_files_per_speaker_in_base_set amount
# of files of each speaker in the total dataset
# ===========================================================
#
n_files = len(speaker_dict.keys()) * num_files_per_speaker_in_base_set
extract_files_into_partition(n_files, '_base_all_speakers')

# Build remaining datasets
# ===========================================================
#
files_remaining = min(map(lambda x: len(x), speaker_dict.values()))
print("Build remaining partitions")
print("\tEach speaker has at least {} files remaining.".format(files_remaining))

partition_counter = 0
while files_remaining > 0:
    files_in_partition = num_files_per_speaker_in_base_set
    if files_remaining < files_in_partition:
        files_in_partition = files_remaining
    
    extract_files_into_partition(len(speaker_dict.keys()) * files_in_partition, '_base_' + str(partition_counter))
    partition_counter += 1
    files_remaining -= files_in_partition
