from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adadelta
from networks.losses import get_loss, add_final_layers


def create_network_n_speakers(num_speakers, config):
    # Read parameters from config
    seg_size = config.getint('pairwise_kldiv', 'seg_size')
    spectrogram_height = config.getint('pairwise_kldiv', 'spectrogram_height')
    lr = config.getfloat('pairwise_kldiv', 'adadelta_learning_rate')
    rho = config.getfloat('pairwise_kldiv', 'adadelta_rho')
    epsilon = config.getfloat('pairwise_kldiv', 'adadelta_epsilon')

    # Initialize model
    model = Sequential()

    # convolution layer 1
    model.add(Conv2D(filters=32, kernel_size=(4, 4), activation='relu',
                     input_shape=(1, seg_size, spectrogram_height), data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))

    # convolution layer 2
    model.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))

    # dense layer
    model.add(Flatten())
    model.add(Dense(units=(num_speakers * 10), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=(num_speakers * 5), activation='relu'))
    add_final_layers(model)

    loss_function = get_loss()

    # Create Optimizer
    adadelta = Adadelta(lr=lr, rho=rho, epsilon=epsilon, decay=0.0)

    # Compile model
    model.compile(loss=loss_function,
                  optimizer=adadelta,
                  metrics=['accuracy'])

    return model
