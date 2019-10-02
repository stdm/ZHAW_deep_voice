# This script can be used to change the indices inside the .h5 partitions from the 
# VoxCeleb2 ActiveLearning (PariwiseLstm Vox2) to their respective number.
#
# This is due to the nature that normally the label indices are from 0 to partition_speaker_num - 1,
# but this leaves all the partiitons to have "the same speakers" what the neural net is concerned.
#
#
# The paritions are as follows:
# - Base Partition                  <name>.h5           Contains the Base N speakers, Index 0 to N-1
# - Additional Partitions           <name>_<index>.h5   Containers N speakers, should have Index from (index+1)*N to (index+1)*N + N-1
#  
#
# Params:
# - base_name: Specifies the name of the partition files
# - base_path: Specifies the path where those partitions lie in
# - partition_size: Specifies the number of speakers per partition, which is used to calculate by how much the indices need to be raised
#
import h5py
import numpy as np

import os

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-base_name", dest="base_name", default='vox2_speakers_10_test_cluster')
parser.add_argument("-base_path", dest="base_path", default='common/data/training/speaker_pickles')
parser.add_argument("-partition_size", dest="partition_size", default=40)

args = parser.parse_args()
base_name = args.base_name
base_path = args.base_path
partition_size = int(args.partition_size)

folder = os.fsencode(base_path)

filenames = []

for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith('.h5') and filename.startswith(base_name):
        partition_ident = filename.split('_')[-1:][0].replace('.h5','')

        if partition_ident == 'cluster':
            pass
            print('Base Partition: Do not change indices')
        else:
            add_to_indices = (int(partition_ident) + 1) * partition_size
            print('')
            print("{}th Partition: Add {} to indices".format(str(partition_ident), str(add_to_indices)))

            with h5py.File(base_path + '/' + filename,'r+') as f:
                y = f['y'][:]
                
                for i in range(len(y)):
                    if y[i] < add_to_indices:
                        f['y'][i] = y[i] + add_to_indices
                        print("Processing done: {} --> {}".format(y[i], f['y'][i]))

                f.close()
