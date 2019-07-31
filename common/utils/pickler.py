import pickle
import h5py

def path_is_pickle(path):
    pickle_format = path.split('.')
    pickle_format = pickle_format[len(pickle_format) - 1]
    return pickle_format == 'pickle'

def save(obj, path):
    if path_is_pickle(path):
        save_pickle(obj, path)
    else:
        save_h5(obj, path)

def load(path):
    if path_is_pickle(path):
        return load_pickle(path)
    else:
        return load_h5(path)

def load_speaker_pickle_or_h5(path):
    if path_is_pickle(path):
        (X, y, speaker_names) = load_pickle(path)
    else:
        with h5py.File(path,'r') as f:
            X = f['X'][:,:,:,:]
            y = f['y'][:]
            speaker_names = f['speaker_names'][:]
            f.close()

    return (X, y, speaker_names)


# Pickles
# -----------------------------------------------------------

# Save a pickle file
def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, -1)

# Load a pickle file
def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj



# H5
# -----------------------------------------------------------

# Saves given tuple to the .h5 with the given path
# 
# (Currently does not work for lists of strings,
# use numpy chararray instead)
# 
def save_h5(my_tuple, path):
    with h5py.File(path, 'w') as f:
        for i in range(len(my_tuple)):
            f.create_dataset(str(i), data=my_tuple[i])
        f.close()

# Load the given .h5 file and returns the contents as tuple
# 
def load_h5(path):
    data = list()

    with h5py.File(path, 'r+') as f:
        for i in range(len(f.keys())):
            data.append(f[str(i)][()])
        f.close()

    return data
