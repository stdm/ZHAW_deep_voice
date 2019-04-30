import pickle
import h5py

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, -1)


def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def load_speaker_pickle_or_h5(path):
    pickle_format = path.split('.')
    pickle_format = pickle_format[len(pickle_format) - 1]

    if pickle_format == 'h5':
        with h5py.File(path,'r') as f:
            X = f['X'][:][:][:][:]
            y = f['y'][:]
            speaker_names = f['speaker_names'][:]
            f.close()
    else:
        with open(path, 'rb') as f:
            (X, y, speaker_names) = pickle.load(f)

    return (X, y, speaker_names)
