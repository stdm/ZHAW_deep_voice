from common.utils.paths import get_result_png
from common.utils.pickler import load
from common.utils.logger import *

import matplotlib.pyplot as plt
import numpy as np

metric_names = ["MR", "ACP", "ARI", "DER"]
metric_worst_values = [1,0,0,1]


def plot_files(plot_file_name, files):
    """
    Plots the results stored in the files given and stores them in a file with the given name
    :param plot_file_name: the file name stored in common/data/results
    :param files: a set of full file paths that hold result data
    """
    curve_names, metric_sets, set_of_number_of_embeddings = _read_result_pickle(files)
    _plot_curves(plot_file_name, curve_names, metric_sets, set_of_number_of_embeddings)


def _read_result_pickle(files):
    """
    Reads the results of a network from these files.
    :param files: can be 1-n files that contain a result.
    :return: curve names, thresholds, metric scores as a list and number of embeddings
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info("Read result pickle")

    # Initialize result sets
    curve_names_all_files = []
    number_of_embeddings_all_files = []
    metric_sets_all_files = [[] for _ in metric_names]

    for file in files:
        #Load results from file
        curve_names, metric_sets, number_of_embeddings = load(file)

        #Add results from file to result sets
        curve_names_all_files.extend(curve_names)
        number_of_embeddings_all_files.extend(number_of_embeddings)
        for m, metric_set in enumerate(metric_sets):
            metric_sets_all_files[m].extend(metric_set)

    return curve_names_all_files, metric_sets_all_files, number_of_embeddings_all_files


def _plot_curves(plot_file_name, curve_names, metric_sets, number_of_embeddings):
    """
    Plots all specified curves and saves the plot into a file.
    :param plot_file_name: String value of save file name
    :param curve_names: Set of names used in legend to describe this curve
    :param metric_sets: A list of 2D matrices, each row of a metrics 2D matrix describes one dataset for a curve
    :param number_of_embeddings: set of integers, each integer describes how many embeddings is in this curve
    """
    maximum_clusters_to_display = number_of_embeddings[0]

    # TODO: Grab config and check if :short_utterances is set
    # ==> If so, only go up to 20% of the maximum clusters
    # ==> Is it always 20%?
    # 
    # if short_utterances?:
    #     maximum_clusters_to_display *= 0.2

    logger = get_logger('analysis', logging.INFO)
    logger.info('Plot results')

    # Slice results to only 1-maximum_clusters_to_display clusters
    for i in range(0,len(metric_sets)):
        for j in range(0, len(metric_sets[i])):
            metric_sets[i][j] = metric_sets[i][j][-maximum_clusters_to_display:]

    best_results = [[] for _ in metric_names]
    for m, min_value in enumerate(metric_worst_values):
        for results in metric_sets[m]:
            if(metric_worst_values[m] == 0):
                best_results[m].append(np.max(results))
            else:
                best_results[m].append(np.min(results))

    # How many lines to plot
    number_of_lines = len(curve_names)

    # Get various colors needed to plot
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(i) for i in np.linspace(0, 1, number_of_lines)]

    # Set fontsize for all plots
    plt.rcParams.update({'font.size': 12})

    # Define number of figures
    fig1 = plt.figure(figsize=(18, 12))

    # Define Plots
    plot_grid = (3, 2)

    plots = [None] * len(metric_names)

    plots[0] = _add_cluster_subplot(plot_grid, (0, 0), metric_names[0], 1, maximum_clusters_to_display)
    plots[1] = _add_cluster_subplot(plot_grid, (0, 1), metric_names[1], 1, maximum_clusters_to_display)
    plots[2] = _add_cluster_subplot(plot_grid, (1, 0), metric_names[2], 1, maximum_clusters_to_display)
    plots[3] = _add_cluster_subplot(plot_grid, (1, 1), metric_names[3], 1, maximum_clusters_to_display)

    # Set the horizontal space between subplots
    plt.subplots_adjust(hspace = 0.3)

    # Define curves and their values
    curves = [[] for _ in metric_names]

    for m, metric_set in enumerate(metric_sets):
        curves[m] = [plots[m], metric_set]

    # Plot all curves
    for index in range(number_of_lines):
        label = curve_names[index]
        for m, metric_name in enumerate(metric_names):
            label = label + '\n {} {}: {}'.format('Max' if metric_worst_values[m]==0 else 'Min', metric_name,
                                                  str(best_results[m][index]))
        color = colors[index]

        number_of_clusters = np.arange(maximum_clusters_to_display, 0, -1)

        line = None
        for plot, value in curves:
            line, = plot.plot(number_of_clusters, value[index], color=color)
            
        line.set_label(label)

    fig1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.33), ncol=4)
    fig1.savefig(get_result_png(plot_file_name + '.png'), format='png')
    fig1.savefig(get_result_png(plot_file_name + '.svg'), format='svg')

    return fig1


def _add_cluster_subplot(grid, position, y_label, colspan=1, x_maximum=80):
    """
    Adds a cluster subplot to the current figure.

    :param grid: a tuple that contains number of rows as the first entry and number of columns as the second entry
    :param position: the position of this subplot
    :param y_label: the label of the y axis
    :param colspan: number of columns for the x axis, default is 1
    :return: the subplot itself
    """
    subplot = plt.subplot2grid(grid, position, colspan=colspan)
    subplot.set_ylabel(y_label)
    subplot.set_xlabel('number of clusters')
    subplot.set_xlim([-3, x_maximum + 3])
    subplot.set_ylim([-0.05, 1.05])
    return subplot
