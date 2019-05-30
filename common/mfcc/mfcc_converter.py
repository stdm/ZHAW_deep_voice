"""
This file opens, converts and compresses the wav file into a usable MFCC for our application.

"""
import librosa

def mfcc(wav_file):
    # Read out audio range and sample rate of wav file
    audio_range, sample_rate = librosa.load(path=wav_file, sr=None)
    hop_length = 256 #this value is based on the KI1-Lab which was used as template for this approach!

    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=audio_range, sr=sample_rate, n_mfcc=20, hop_length=hop_length, n_fft=1024)


    return mfcc #return the features in the format expected by scikit-learn
