from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist

from common.utils.logger import *


def cluster_embeddings(set_of_embeddings, metric='cosine', method='complete'):
    """
    Calculates the distance and the linkage matrix for these embeddings.

    :param set_of_embeddings: The embeddings we want to calculate on
    :param metric: The metric used for the distance and linkage
    :param method: The linkage method used.
    :return: The embedding Distance and the embedding linkage
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Cluster embeddings')

    set_predicted_clusters = []

    for embeddings in set_of_embeddings:
        embeddings_distance = cdist(embeddings, embeddings, metric)
        embeddings_linkage = linkage(embeddings_distance, method, metric)

        thresholds = embeddings_linkage[:, 2]
        predicted_clusters = []

        for threshold in thresholds:
            predicted_cluster = fcluster(embeddings_linkage, threshold, 'distance')
            predicted_clusters.append(predicted_cluster)

        set_predicted_clusters.append(predicted_clusters)

    return set_predicted_clusters
