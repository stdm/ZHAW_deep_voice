from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
import common.dominant_sets.dominantset as ds
import numpy as np

from common.utils.logger import *


def cluster_embeddings(set_of_embeddings, set_of_true_clusters, dominant_sets=False,
                       metric='cosine', method='complete'):
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

    for embeddings, true_clusters in zip(set_of_embeddings, set_of_true_clusters):
        if dominant_sets:
            predicted_clusters = []
            dos = ds.DominantSetClustering(feature_vectors=np.array(embeddings),
                                           speaker_ids=np.array(true_clusters),
                                           metric='cosine', dominant_search=False,
                                           epsilon=1e-6, cutoff=0.125)
            dos.apply_clustering()
            predicted_clusters.append(dos.ds_result)
        else:
            predicted_clusters = original_clustering(embeddings, metric, method)

        set_predicted_clusters.append(predicted_clusters)

    return set_predicted_clusters


def original_clustering(embeddings, metric, method):
    embeddings_distance = cdist(embeddings, embeddings, metric)
    embeddings_linkage = linkage(embeddings_distance, method, metric)

    thresholds = embeddings_linkage[:, 2]
    predicted_clusters = []

    for threshold in thresholds:
        predicted_cluster = fcluster(embeddings_linkage, threshold, 'distance')
        predicted_clusters.append(predicted_cluster)

    return predicted_clusters




