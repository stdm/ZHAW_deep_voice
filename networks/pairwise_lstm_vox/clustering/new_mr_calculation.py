import operator
import pickle
from operator import itemgetter

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist

from common.utils.paths import *


def cluster_embeddings(embeddings):
    """Caluclates Linkage matrix from embeddings.
    """
    embeddings_dist = cdist(embeddings, embeddings, 'cosine')
    embeddings_linkage = linkage(embeddings_dist, method='complete', metric='cosine')

    return embeddings_dist, embeddings_linkage


def calculate_minimal_mr(embeddings_linkage, map_labels):
    """Caluclates minimal missclassification rate with given linkage matrix of embeddings.
    """
    thresholds = embeddings_linkage[:, 2]
    maps_clusters = []
    mrs = np.ones((thresholds.shape))

    # Loop over all possible clusterings
    for i, threshold in enumerate(thresholds):
        map_clusters = fcluster(embeddings_linkage, threshold, 'distance')
        maps_clusters.append(map_clusters)

        # MR Calculation for this threshold
        mrs[i] = calculate_mr(map_clusters, map_labels)

    # Return minimal MR and the according cluster map and threshold
    mr = np.min(mrs)
    map_clusters = maps_clusters[np.argmin(mrs)]
    threshold = thresholds[np.argmin(mrs)]

    return mr, map_clusters, threshold


def calculate_mr(map_clusters, map_labels):
    """Calculates MR for a given clustering
    """
    number_of_clusters = len(set(map_clusters))
    number_of_labels = len(set(map_labels))
    number_of_segments = len(map_labels)
    # Data structure to assign best speaker to cluster

    # Build dict with count of occurrences of samples per speaker per cluster
    # cluster no -> label no -> occurrences
    cluster_label_count = {}
    # Build dict with count of occurrences of samples per cluster per speaker
    # label no -> cluster no -> occurrences (inverted index)
    label_cluster_count = {}

    for cluster in set(map_clusters):
        cluster = cluster.item()
        cluster_label_count[cluster] = {}
        for label_no, cluster_no in zip(map_labels, map_clusters):
            label_no = label_no.item()
            cluster_no = cluster_no.item()
            if cluster_no == cluster:
                # cluster_label_count
                if label_no in cluster_label_count[cluster]:
                    cluster_label_count[cluster][label_no] += 1
                else:
                    cluster_label_count[cluster][label_no] = 1

                # label_cluster_count
                if label_no not in label_cluster_count:
                    label_cluster_count[label_no] = {}
                if cluster in label_cluster_count[label_no]:
                    label_cluster_count[label_no][cluster] += 1
                else:
                    label_cluster_count[label_no][cluster] = 1

    # print cluster_label_count
    # print label_cluster_count
    # Assign one speaker per cluster
    # Speaker with most samples in cluster, if no other cluster has more samples of the same speaker
    # Order of occurends in case of even situations, cluster with -1 are without assignment
    cluster_label = [-1 for i in range(number_of_clusters)]
    for label_no in label_cluster_count:
        # Sorted list of occurrences in various clusters
        occurrences = sorted(label_cluster_count[label_no].items(), key=operator.itemgetter(1))
        occurrences.reverse()

        # Loop to find max for speaker and cluster
        for occurrence in occurrences:
            # Check if current speaker is the dominant speaker in cluster
            if max(cluster_label_count[occurrence[0]], key=cluster_label_count[occurrence[0]].get) == label_no:
                cluster_label[occurrence[0] - 1] = label_no
                break

    # Count missclassified segments
    missclassified_segments = 0
    for cluster in cluster_label_count:
        for speaker in cluster_label_count[cluster]:
            if speaker != cluster_label[cluster - 1]:
                print(speaker)
                print(cluster)
                missclassified_segments += cluster_label_count[cluster][speaker]

    return (1. / number_of_segments) * missclassified_segments


def calculate_legacy_minimal_mr(embeddings_linkage, y, num_speakers):
    """MR calculation of Lukic et al.
    """
    clusters = []
    for i in range(len(y)):
        clusters.append([i])

    for z in embeddings_linkage:
        clusters.append(clusters[int(z[0])] + clusters[int(z[1])])

    e = []
    e.append(np.ones(len(y), dtype=np.int))

    for z in embeddings_linkage:
        err = list(e[len(e) - 1])
        idx1 = int(z[0])
        idx2 = int(z[1])
        if idx1 >= len(y) or idx2 >= len(y) or y[idx1] != y[idx2]:
            indices = clusters[idx1] + clusters[idx2]
            increase_error(indices, err, clusters)
        else:
            err[idx1] = 0
            err[idx2] = 0
        e.append(err)

    MRs = []
    for err in e:
        MRs.append(float(sum(err)) / len(y))

    # print('MR={:.4f}'.format(np.min(MRs)))
    return np.min(MRs)


def increase_error(indices, e, clusters):
    for i in indices:
        if i < len(e):
            e[i] = 1
        else:
            increase_error(clusters[i], e, clusters)


def generate_X(train_output, test_output, train_speakers, test_speakers, neuron_number):
    num_speakers = len(set(test_speakers))
    # print num_speakers
    X_train, y_train = extract_vectors(num_speakers, neuron_number, train_output, train_speakers)
    X_test, y_test = extract_vectors(num_speakers, neuron_number, test_output, test_speakers)
    X = []
    X.extend(X_train)
    X.extend(X_test)
    y = []
    y.extend(y_train)
    y.extend(y_test)
    return X, y, num_speakers


def extract_vectors(num_speakers, vec_size, vectors, y):
    X = np.zeros((num_speakers, vec_size))
    for i in range(num_speakers):
        indices = np.where(y == i)[0]
        outputs = np.take(vectors, indices, axis=0)
        for o in outputs:
            X[i] = np.add(X[i], o)
        X[i] = np.divide(X[i], len(outputs))
    return X, set(y)


def print_cluster(map_cluster, y, speakers):
    clust = []
    for i in range(len(y)):
        # print map_cluster[i]
        clust.append((map_cluster[i], speakers[y[i]]))

    s_clust = sorted(clust, key=itemgetter(0))

    for c in s_clust:
        print(c)
    print(len(clust))


if __name__ == "__main__":
    with open(get_speaker_pickle('cluster_output_train_40sp_2017-05-25_21-23-42'), 'rb') as f:
        cluster_output, speakers, speaker_names = pickle.load(f)

    with open(get_speaker_pickle('cluster_output_test_40sp_2017-05-25_21-23-42'), 'rb') as f:
        cluster_output1, speakers1, speaker_names1 = pickle.load(f)

    # print cluster_output1.shape
    # print cluster_output.shape
    # print speakers
    # print speakers1.shape


    X, y, num_speakers = generate_X(cluster_output, cluster_output1, speakers, speakers1, 256)
    _, emb = cluster_embeddings(X)
    mr, map_c, th = calculate_minimal_mr(emb, y)
    print(mr)
    print(th)
    # print map_c
    # print y[29]
    # print y[29+40]
    # print map_c[29]
    # print map_c[29+40]
    print_cluster(map_c, y, speaker_names)

    print("understand mr calc")
    calculate_mr(map_c, y)
