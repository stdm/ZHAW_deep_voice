# Data Configs
ONE_SEC = 100  # spectrogram width
INTERVALL = 15  # input data width
FREQ_ELEMENTS = 128  # spectrogram height
SENTENCES_PER_SPEAKER = 10
SENTENCES_TRAIN = 8
NUM_OF_SPEAKERS = 630

# ArcFace
MARGIN_S = 1.0
MARGIN_M1 = 1.0
MARGIN_M2 = 0.5
MARGIN_M3 = 0.0

# Network Configs
LSTM_HIDDEN_1 = 256
LSTM_HIDDEN_2 = 256
DENSE_HIDDEN_1 = 10
DENSE_HIDDEN_2 = 5
DROP_RATE_1 = 0.5
DROP_RATE_2 = 0.25

# Controller Configs
MAX_EPOCHS = 10
BATCH_SIZE = 256
BATCHES_PER_EPOCH = 100
TRAIN_DATA_NAME = 'speakers_100_50w_50m_not_reynolds_cluster'
VAL_DATA_NAME = 'speakers_40_clustering_vs_reynolds'
