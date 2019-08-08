from common.analysis.metrics.acp import average_cluster_purity
from common.analysis.metrics.ari import adjusted_rand_index
from common.analysis.metrics.der import diarization_error_rate

import numpy as np

from common.analysis.metrics.mr import misclassification_rate
from common.utils.logger import *
from common.utils.paths import *
from common.utils.pickler import load, save

metric_names = ["MR", "ACP", "ARI", "DER"]
metric_worst_values = [1,0,0,1]


def analyse_results(network_name, checkpoint_names, set_of_predicted_clusters,
                    set_of_true_clusters, embedding_numbers, set_of_times):
    """
    Analyses each checkpoint with the values of set_of_predicted_clusters and set_of_true_clusters.
    After the analysis the result are stored in the Pickle network_name.pickle and the best Result
    according to min MR is stored in network_name_best.pickle.
    :param network_name: The name for the result pickle.
    :param checkpoint_names: A list of names from the checkpoints. Later used as curvenames,
    :param set_of_predicted_clusters: A 2D array of the predicted Clusters from the Network. [checkpoint, clusters]
    :param set_of_true_clusters: A 2d array of the validation clusters. [checkpoint, validation-clusters]
    :param embedding_numbers: A list which represent the number of embeddings in each checkpoint.
    :param set_of_times: A 2d array of the time per utterance [checkpoint, times]
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Run analysis')
    metric_sets = [[None] * len(set_of_predicted_clusters) for _ in range(len(metric_names))]

    for index, predicted_clusters in enumerate(set_of_predicted_clusters):
        checkpoint = checkpoint_names[index]
        logger.info('Analysing checkpoint:' + checkpoint)

        # Check if checkpoint is already stored
        analysis_pickle = get_results_intermediate_analysis(checkpoint)

        if os.path.isfile(analysis_pickle):
            metric_results = load(analysis_pickle)
        else:
            metric_results = _calculate_analysis_values(predicted_clusters, set_of_true_clusters[index], set_of_times[index])
            save(metric_results, analysis_pickle)

        for m, metric_result in enumerate(metric_results):
            metric_sets[m][index] = metric_result

    _write_result_pickle(network_name, checkpoint_names, metric_sets, embedding_numbers)
    _save_best_results(network_name, checkpoint_names, metric_sets, embedding_numbers)

    logger.info('Clearing intermediate result checkpoints')
    
    for checkpoint in checkpoint_names:
        analysis_pickle = get_results_intermediate_analysis(checkpoint)
        test_pickle = get_results_intermediate_test(checkpoint)

        if os.path.exists(analysis_pickle):
            os.remove(analysis_pickle)

        if os.path.exists(test_pickle):
            os.remove(test_pickle)

    logger.info('Analysis done')


def _calculate_analysis_values(predicted_clusters, true_cluster, times):
    """
    Calculates the analysis values out of the predicted_clusters.

    :param predicted_clusters: The predicted Clusters of the Network.
    :param true_clusters: The validation clusters
    :return: the results of all metrics as a 2D array where i is the index of the metric and j is the index of a
        specific result

    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Calculate scores')

    # Initialize output
    metric_results = [None] * len(metric_names)
    for m, min_value in enumerate(metric_worst_values):
        if min_value == 1:
            metric_results[m] = np.ones(len(true_cluster))
        else:
            metric_results[m] = np.zeros((len(true_cluster)))

    # Loop over all possible clustering
    for i, predicted_cluster in enumerate(predicted_clusters):
        logger.info('Calculated Scores for {}/{} predicted clusters'.format(i, len(predicted_clusters)))
        # Calculate different analysis's
        metric_results[0][i] = misclassification_rate(true_cluster, predicted_cluster)
        metric_results[1][i] = average_cluster_purity(true_cluster, predicted_cluster)
        metric_results[2][i] = adjusted_rand_index(true_cluster, predicted_cluster)
        metric_results[3][i] = diarization_error_rate(true_cluster, predicted_cluster, times)

    return metric_results


def _save_best_results(network_name, checkpoint_names, metric_sets, speaker_numbers):
    if len(metric_sets[0]) == 1:
        _write_result_pickle(network_name + "_best", checkpoint_names, metric_sets, speaker_numbers)
    else:
        # Find best result (according to the first metric in metrics)
        if metric_worst_values[0] == 1:
            best_results = []
            for results in metric_sets[0]:
                best_results.append(np.min(results))
            best_result_over_all = min(best_results)
        else:
            best_results = []
            for results in metric_sets[0]:
                best_results.append(np.max(results))
            best_result_over_all = max(best_results)

        best_checkpoint_name = []
        set_of_best_metrics = [[] for _ in metric_sets]
        best_speaker_numbers = []

        for index, best_result in enumerate(best_results):
            if best_result == best_result_over_all:
                best_checkpoint_name.append(checkpoint_names[index])
                for m, metric_set in enumerate(metric_sets):
                    set_of_best_metrics[m].append(metric_set[index])
                best_speaker_numbers.append(speaker_numbers[index])

        _write_result_pickle(network_name + "_best", best_checkpoint_name, set_of_best_metrics, best_speaker_numbers)


def _write_result_pickle(network_name, checkpoint_names, metric_sets, number_of_embeddings):
    logger = get_logger('analysis', logging.INFO)
    logger.info('Write result pickle')
    save((checkpoint_names, metric_sets, number_of_embeddings), get_result_pickle(network_name))
