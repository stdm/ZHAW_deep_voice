from keras import Model
from keras.engine.saving import load_model

from common.clustering.generate_embeddings import generate_embeddings
from common.utils import TimeCalculator
from common.utils.logger import *
from common.utils.paths import get_experiment_nets
from networks.losses import get_custom_objects, get_loss

import numpy as np


logger = get_logger('kldiv', logging.INFO)


def create_embeddings(config, checkpoints, x_list, y_list, out_layer=7, seg_size=100):
    # Prepare return value
    set_of_embeddings = []
    set_of_speakers = []
    set_of_num_embeddings = []
    set_of_total_times = []

    # Values out of the loop
    metrics = ['accuracy']
    loss = get_loss(config)
    custom_objects = get_custom_objects(config)
    optimizer = 'adadelta'

    for checkpoint in checkpoints:
        logger.info('Run checkpoint: ' + checkpoint)
        # Load and compile the trained network
        network_file = get_experiment_nets(checkpoint)
        model_full = load_model(network_file, custom_objects=custom_objects)
        model_full.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # Get a Model with the embedding layer as output and predict
        model_partial = Model(inputs=model_full.input, outputs=model_full.layers[out_layer].output)

        x_cluster_list = []
        y_cluster_list = []
        for x, y in zip(x_list, y_list):
            x_cluster = np.asarray(model_partial.predict(x))
            x_cluster_list.append(x_cluster)
            y_cluster_list.append(y)

        embeddings, speakers, num_embeddings = \
            generate_embeddings(x_cluster_list, y_cluster_list, x_cluster_list[0].shape[1])

        # Fill return values
        set_of_embeddings.append(embeddings)
        set_of_speakers.append(speakers)
        set_of_num_embeddings.append(num_embeddings)

        # Calculate the time per utterance
        time = TimeCalculator.calc_time_all_utterances(y_cluster_list, seg_size)
        set_of_total_times.append(time)

    return checkpoints, set_of_embeddings, set_of_speakers, set_of_num_embeddings, set_of_total_times
