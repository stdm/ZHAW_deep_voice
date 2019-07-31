import numpy as np

from common.utils.logger import *


def generate_embeddings(outputs, speakers_inputs, vector_size):
    """
    Combines the utterances of the speakers in the train- and testing-set and combines them into embeddings.
    :param train_output: The training output (8 sentences)
    :param test_output:  The testing output (2 sentences)
    :param train_speakers: The speakers used in training
    :param test_speakers: The speakers used in testing
    :param vector_size: The size which the output will have
    :return: embeddings, the speakers and the number of embeddings
    """
    logger = get_logger('clustering', logging.INFO)
    logger.info('Generate embeddings')
    num_speakers = len(set(speakers_inputs[0]))

    # Prepare return variable
    number_embeddings = len(outputs) * num_speakers
    embeddings = []
    speakers = []

    for output, speakers_input in zip(outputs, speakers_inputs):
        embeddings_output, speakers_output = _create_utterances(num_speakers, vector_size, output, speakers_input)
        embeddings.extend(embeddings_output)
        speakers.extend(speakers_output)

    return embeddings, speakers, number_embeddings


def _create_utterances(num_speakers, vector_size, vectors, y):
    """
    Creates one utterance for each speaker in the vectors.
    :param num_speakers: Number of distinct speakers in this vector
    :param vector_size: Number of data in utterance
    :param vectors: The unordered speaker data
    :param y: An array that tells which speaker (number) is in which place of the vectors array
    :return: the embeddings per speaker and the speakers (numbers)
    """

    # Prepare return variables
    embeddings = np.zeros((num_speakers, vector_size))
    speakers = set(y)

    # Fill embeddings with utterances
    for i in range(num_speakers):

        # Fetch correct utterance
        utterance = embeddings[i]

        # Fetch values where same speaker and add to utterance
        indices = np.where(y == i)[0]
        
        outputs = np.take(vectors, indices, axis=0)
        for value in outputs:
            utterance = np.add(utterance, value)

        # Add filled utterance to embeddings
        embeddings[i] = np.divide(utterance, len(outputs))

    return embeddings, speakers
