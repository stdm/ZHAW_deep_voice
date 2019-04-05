import matplotlib

from common.analysis.acp import average_cluster_purity
from common.analysis.ari import adjusted_rand_index

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import *
from theano.gradient import np

from common.analysis.mr import misclassification_rate
from common.utils.logger import *
from common.utils.paths import *
from common.utils.pickler import load, save


def plot_files(plot_file_name, files):
    """
    Plots the results stored in the files given and stores them in a file with the given name
    :param plot_file_name: the file name stored in common/data/results
    :param files: a set of full file paths that hold result data
    """
    curve_names, set_of_mrs, set_of_acps, set_of_aris, set_of_homogeneity_scores, \
        set_of_completeness_scores, set_of_number_of_embeddings = _read_result_pickle(files)

    #Todo: Plot acps and aris
    _plot_curves(plot_file_name, curve_names, set_of_mrs, set_of_homogeneity_scores, set_of_completeness_scores,
                set_of_number_of_embeddings)


def _read_result_pickle(files):
    """
    Reads the results of a network from these files.
    :param files: can be 1-n files that contain a result.
    :return: curve names, thresholds, mrs, homogeneity scores, completeness scores and number of embeddings
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Read result pickle')
    curve_names = []

    # Initialize result sets
    set_of_thresholds = []
    set_of_mrs = []
    set_of_acps = []
    set_of_aris = []
    set_of_homogeneity_scores = []
    set_of_completeness_scores = []
    set_of_number_of_embeddings = []

    # Fill result sets
    for file in files:
        curve_name, mrs, acps, aris, homogeneity_scores, completeness_scores, number_of_embeddings = load(file)

        for index, curve_name in enumerate(curve_name):
            set_of_mrs.append(mrs[index])
            set_of_acps.append(acps[index])
            set_of_aris.append(aris[index])
            set_of_homogeneity_scores.append(homogeneity_scores[index])
            set_of_completeness_scores.append(completeness_scores[index])
            set_of_number_of_embeddings.append(number_of_embeddings[index])
            curve_names.append(curve_name)

    return curve_names, set_of_mrs, set_of_acps, set_of_aris, set_of_homogeneity_scores, set_of_completeness_scores, set_of_number_of_embeddings


def _plot_curves(plot_file_name, curve_names, mrs, homogeneity_scores, completeness_scores, number_of_embeddings):
    """
    Plots all specified curves and saves the plot into a file.
    :param plot_file_name: String value of save file name
    :param curve_names: Set of names used in legend to describe this curve
    :param mrs: 2D Matrix, each row describes one dataset of misclassification rates for a curve
    :param homogeneity_scores: 2D Matrix, each row describes one dataset of homogeneity scores for a curve
    :param completeness_scores: 2D Matrix, each row describes one dataset of completeness scores for a curve
    :param number_of_embeddings: set of integers, each integer describes how many embeddings is in this curve
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Plot results')
    min_mrs = []
    for mr in mrs:
        min_mrs.append(np.min(mr))

    min_mrs, curve_names, mrs, homogeneity_scores, completeness_scores, number_of_embeddings = \
        (list(t) for t in
         zip(*sorted(zip(min_mrs, curve_names, mrs, homogeneity_scores, completeness_scores, number_of_embeddings))))

    # How many lines to plot
    number_of_lines = len(curve_names)

    # Get various colors needed to plot
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(i) for i in np.linspace(0, 1, number_of_lines)]

    # Define number of figures
    fig1 = plt.figure(1)
    fig1.set_size_inches(16, 8)

    # Define Plots
    mr_plot = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    mr_plot.set_ylabel('MR')
    mr_plot.set_xlabel('number of clusters')
    plt.ylim([-0.02, 1.02])

    completeness_scores_plot = _add_cluster_subplot(fig1, 234, 'completeness_scores')
    homogeneity_scores_plot = _add_cluster_subplot(fig1, 235, 'homogeneity_scores')

    # Define curves and their values
    curves = [[mr_plot, mrs],
              [homogeneity_scores_plot, homogeneity_scores],
              [completeness_scores_plot, completeness_scores]]

    # Plot all curves
    for index in range(number_of_lines):
        label = curve_names[index] + '\n min MR: ' + str(min_mrs[index])
        color = colors[index]
        number_of_clusters = np.arange(number_of_embeddings[index], 0, -1)

        for plot, value in curves:
            plot.plot(number_of_clusters, value[index], color=color, label=label)

    # Add legend and save the plot
    fig1.legend()
    # fig1.show()
    fig1.savefig(get_result_png(plot_file_name))
    fig1.savefig(get_result_png(plot_file_name + '.svg'), format='svg')


def _add_cluster_subplot(fig, position, y_label):
    """
    Adds a cluster subplot to the given figure.

    :param fig: the figure which gets a new subplot
    :param position: the position of this subplot
    :param title: the title of the subplot
    :return: the subplot itself
    """
    subplot = fig.add_subplot(position)
    subplot.set_ylabel(y_label)
    subplot.set_xlabel('number of clusters')
    return subplot


def analyse_results(network_name, checkpoint_names, set_of_predicted_clusters, set_of_true_clusters, embedding_numbers):
    """
    Analyses each checkpoint with the values of set_of_predicted_clusters and set_of_true_clusters.
    After the analysis the result are stored in the Pickle network_name.pickle and the best Result
    according to min MR is stored in network_name_best.pickle.

    :param network_name: The name for the result pickle.
    :param checkpoint_names: A list of names from the checkpoints. Later used as curvenames,
    :param set_of_predicted_clusters: A 2D array of the predicted Clusters from the Network. [checkpoint, clusters]
    :param set_of_true_clusters: A 2d array of the validation clusters. [checkpoint, validation-clusters]
    :param embeddings_numbers: A list which represent the number of embeddings in each checkpoint.
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Run analysis')
    set_of_mrs = []
    set_of_acps = []
    set_of_aris = []
    set_of_homogeneity_scores = []
    set_of_completeness_scores = []

    for index, predicted_clusters in enumerate(set_of_predicted_clusters):
        logger.info('Analysing checkpoint:' + checkpoint_names[index])

        mrs, acps, aris, homogeneity_scores, completeness_scores = _calculate_analysis_values(predicted_clusters,
                                                                                 set_of_true_clusters[index])
        set_of_mrs.append(mrs)
        set_of_acps.append(acps)
        set_of_aris.append(aris)
        set_of_homogeneity_scores.append(homogeneity_scores)
        set_of_completeness_scores.append(completeness_scores)

    _write_result_pickle(network_name, checkpoint_names, set_of_mrs, set_of_acps, set_of_aris,
                        set_of_homogeneity_scores, set_of_completeness_scores, embedding_numbers)
    _save_best_results(network_name, checkpoint_names, set_of_mrs, set_of_acps, set_of_aris,
                      set_of_homogeneity_scores, set_of_completeness_scores, embedding_numbers)
    logger.info('Analysis done')


def _calculate_analysis_values(predicted_clusters, true_cluster):
    """
    Calculates the analysis values out of the predicted_clusters.

    :param predicted_clusters: The predicted Clusters of the Network.
    :param true_clusters: The validation clusters
    :return: misclassification rate, homogeneity Score, completeness score and the thresholds.
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Calculate scores')

    # Initialize output
    mrs = np.ones(len(true_cluster))
    acps = np.zeros(len(true_cluster))
    aris = np.zeros(len(true_cluster))
    homogeneity_scores = np.ones(len(true_cluster))
    completeness_scores = np.ones(len(true_cluster))

    # Loop over all possible clustering
    for i, predicted_cluster in enumerate(predicted_clusters):
        # Calculate different analysis's
        mrs[i] = misclassification_rate(true_cluster, predicted_cluster)
        acps[i] = average_cluster_purity(true_cluster, predicted_cluster)
        aris[i] = adjusted_rand_index(true_cluster, predicted_cluster)
        homogeneity_scores[i] = homogeneity_score(true_cluster, predicted_cluster)
        completeness_scores[i] = completeness_score(true_cluster, predicted_cluster)

    return mrs, acps, aris, homogeneity_scores, completeness_scores


def _save_best_results(network_name, checkpoint_names, set_of_mrs, set_of_acps, set_of_aris,
                      set_of_homogeneity_scores, set_of_completeness_scores, speaker_numbers):
    if len(set_of_mrs) == 1:
        _write_result_pickle(network_name + "_best", checkpoint_names, set_of_mrs, set_of_acps, set_of_aris,
                            set_of_homogeneity_scores, set_of_completeness_scores, speaker_numbers)
    else:

        # Find best result (min MR)
        min_mrs = []
        for mrs in set_of_mrs:
            min_mrs.append(np.min(mrs))

        min_mr_over_all = min(min_mrs)

        best_checkpoint_name = []
        set_of_best_mrs = []
        set_of_best_acps = []
        set_of_best_aris = []
        set_of_best_homogeneity_scores = []
        set_of_best_completeness_scores = []
        best_speaker_numbers = []
        for index, min_mr in enumerate(min_mrs):
            if min_mr == min_mr_over_all:
                best_checkpoint_name.append(checkpoint_names[index])
                set_of_best_mrs.append(set_of_mrs[index])
                set_of_best_acps.append(set_of_acps[index])
                set_of_best_aris.append(set_of_aris[index])
                set_of_best_homogeneity_scores.append(set_of_homogeneity_scores[index])
                set_of_best_completeness_scores.append(set_of_completeness_scores[index])
                best_speaker_numbers.append(speaker_numbers[index])

        _write_result_pickle(network_name + "_best", best_checkpoint_name,
                            set_of_best_mrs, set_of_best_acps, set_of_best_aris,
                            set_of_best_homogeneity_scores, set_of_best_completeness_scores, best_speaker_numbers)


def _write_result_pickle(network_name, checkpoint_names, set_of_mrs, set_of_acps, set_of_aris,
                        set_of_homogeneity_scores, set_of_completeness_scores, number_of_embeddings):
    logger = get_logger('analysis', logging.INFO)
    logger.info('Write result pickle')
    save((checkpoint_names, set_of_mrs, set_of_acps, set_of_aris, set_of_homogeneity_scores,
          set_of_completeness_scores, number_of_embeddings), get_result_pickle(network_name))


def _read_and_safe_best_results():
    checkpoint_names, set_of_mrs, set_of_acps, set_of_aris,\
        set_of_homogeneity_scores, set_of_completeness_scores, speaker_numbers =\
        _read_result_pickle([get_result_pickle('flow_me')])
    _save_best_results('flow_me', checkpoint_names, set_of_mrs, set_of_acps, set_of_aris,
                       set_of_homogeneity_scores, set_of_completeness_scores, speaker_numbers)


if __name__ == '__main__':
    _read_and_safe_best_results()
