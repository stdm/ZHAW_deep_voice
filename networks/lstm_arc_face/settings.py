# Data Configs
ONE_SEC = 100  # spectrogram width
INTERVALL = 15  # input data width
FREQ_ELEMENTS = 128  # spectrogram height
SENTENCES_PER_SPEAKER = 10
SENTENCES_TRAIN = 8
NUM_OF_SPEAKERS = 630

# ArcFace Configs

# Controller Configs
MAX_EPOCHS = 100
BATCH_SIZE = 256
BATCHES_PER_EPOCH = 10
TRAIN_DATA_NAME = 'speakers_100_50w_50m_not_reynolds_cluster'
VAL_DATA_NAME = 'speakers_40_clustering_vs_reynolds'
