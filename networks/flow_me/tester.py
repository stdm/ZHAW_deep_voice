# !/usr/bin/env python

import os

import matplotlib

matplotlib.use('Agg')

import networks.flow_me.DataGen as DataGen
from common.analysis.old_cluster import *
from common.analysis.old_colors import *
from common.analysis.old_plots import *
from .nn.restore_session import *
from common.utils.load_config import *
from common.utils.logger import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Filter warnings out


def test(checkpoint, list, configPath, nchw):
    path_master_config = './config/master.cfg'
    config = load_config(path_master_config, configPath)
    logger = get_logger('cluster', logging.INFO)

    list = 'test_list' + list.__str__()

    output_layer_name = config.get('test', 'output_layer')

    # Output folder
    output_layer_name = output_layer_name.replace('\\', '/')
    # output_layer_name = output_layer_name.replace(':', '')
    output_layer_name_folder = output_layer_name.replace('\\', '-')  # NOTE: Windows modification, change for deploy
    output_layer_name_folder = output_layer_name_folder.replace(':', '')

    output_folder = os.path.join(config.get('output', 'plot_path'), config.get('exp', 'name'),
                                 output_layer_name_folder, 'ckpt_{:05d}'.format(checkpoint), config.get(list, 'name'))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    add_file_handler(logger, os.path.join(output_folder, 'mr.log'))

    # Load Session
    path_meta_graph = os.path.join(config.get('output', 'log_path'), 'train_' + config.get('exp', 'name'),
                                   'model.ckpt-' + str(checkpoint) + '.meta')
    path_ckpt = os.path.join(config.get('output', 'log_path'), 'train_' + config.get('exp', 'name'),
                             'model.ckpt-' + str(checkpoint))

    sess, input_layer, output_layer, train_mode = restore_session(path_meta_graph, path_ckpt, output_layer_name)

    # Load test data
    data_gen_8 = DataGen.DataGen(config, data_set=list, pickle_no='pickle1')
    data_gen_2 = DataGen.DataGen(config, data_set=list, pickle_no='pickle2')

    input_data_8, map_labels_8 = data_gen_8.get_timit_test_set(sentence_pickle=config.getint(list, 'sentences_pickle1'))
    input_data_2, map_labels_2 = data_gen_2.get_timit_test_set(sentence_pickle=config.getint(list, 'sentences_pickle2'))

    labels = data_gen_8.get_labels()  # Both data_gen instances should return the same labels

    if not nchw:
        input_data_8 = np.transpose(input_data_8, axes=(0, 2, 3, 1))
        input_data_2 = np.transpose(input_data_2, axes=(0, 2, 3, 1))

        batch_size = config.getint('net', 'batch_size')
        input_data_8 = input_data_8[0:batch_size][:][:][:]
        input_data_2 = input_data_2[0:batch_size][:][:][:]

        # (858, 128, 100, 1)
        # (?, 128, 100, 1)
        print(np.shape(input_data_8))
        print(np.shape(input_data_2))

    label_colors = get_colors(len(labels))

    # Pass samples trough network, get embeddings
    embeddings_8 = sess.run(output_layer, feed_dict={input_layer: input_data_8, train_mode: False})
    embeddings_2 = sess.run(output_layer, feed_dict={input_layer: input_data_2, train_mode: False})
    sess.close()

    # Reduce to mean, merge to one list of embeddings
    # Init data structures according to used embeddings dimension
    if 'l10_dense' in config.get('test', 'output_layer'):
        embeddings_mean_8 = np.zeros((config.getint(list, 'total_speakers'), int(config.getfloat(
            'net', 'dense10_factor') * config.getint('train', 'total_speakers'))))
        embeddings_mean_2 = np.zeros((config.getint(list, 'total_speakers'), int(config.getfloat(
            'net', 'dense10_factor') * config.getint('train', 'total_speakers'))))
    elif 'l11_dense' in config.get('test', 'output_layer'):
        embeddings_mean_8 = np.zeros((config.getint(list, 'total_speakers'), int(config.getfloat(
            'net', 'dense11_factor') * config.getint('train', 'total_speakers'))))
        embeddings_mean_2 = np.zeros((config.getint(list, 'total_speakers'), int(config.getfloat(
            'net', 'dense11_factor') * config.getint('train', 'total_speakers'))))
    else:
        embeddings_mean_8 = np.zeros((config.getint(list, 'total_speakers'), int(config.getfloat(
            'net', 'dense7_factor') * config.getint('train', 'total_speakers'))))
        embeddings_mean_2 = np.zeros((config.getint(list, 'total_speakers'), int(config.getfloat(
            'net', 'dense7_factor') * config.getint('train', 'total_speakers'))))

    # Sum of all embeddings
    for embedding, label in zip(embeddings_8, map_labels_8):
        embeddings_mean_8[label] += embedding
    for embedding, label in zip(embeddings_2, map_labels_2):
        embeddings_mean_2[label] += embedding

    # Divsion trough count of embeddings from the same speaker
    unique_8, counts_8 = np.unique(map_labels_8, return_counts=True)
    for label, count in zip(unique_8, counts_8):
        embeddings_mean_8[label] /= count

    unique_2, counts_2 = np.unique(map_labels_2, return_counts=True)
    for label, count in zip(unique_2, counts_2):
        embeddings_mean_2[label] /= count

    # Concatenate to one date set for analysis
    embeddings = np.concatenate((embeddings_mean_8, embeddings_mean_2))
    map_labels = np.concatenate((unique_8, unique_2))

    map_labels_full = []

    for label_no in map_labels:
        map_labels_full.append(labels[label_no])

    # Cluster embeddings
    embeddings += 0.00001  # hotfix for division by zero = nan
    embeddings_dist, embeddings_linkage = cluster_embeddings(embeddings)

    # Dendogram
    if config.getint(list, 'total_speakers') > 10:
        draw_dendogram(embeddings_linkage, embeddings_dist, labels, map_labels, map_labels_full, label_colors,
                       figsize=(20, 4.8))
    elif config.getint(list, 'total_speakers') > 40:
        draw_dendogram(embeddings_linkage, embeddings_dist, labels, map_labels, map_labels_full, label_colors,
                       figsize=(50, 4.8))
    else:
        draw_dendogram(embeddings_linkage, embeddings_dist, labels, map_labels, map_labels_full, label_colors)

    plt.savefig(os.path.join(output_folder, 'dendo.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, 'dendo.svg'), bbox_inches='tight')

    axes = plt.gca()
    axes.set_ylim([0, 0.1])
    plt.yticks(np.arange(0, 0.1, 0.005))
    plt.grid(axis='y', linestyle='dotted')
    plt.draw()
    plt.savefig(os.path.join(output_folder, 'dendo_0.1.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, 'dendo_0.1.svg'), bbox_inches='tight')

    # t-SNE Plot
    ''' Take out tsne because of sklearn misbehaving
    if config.getint(list, 'total_speakers') > 20:
        legend = False
    else:
        legend = True

    draw_t_sne(embeddings, labels, map_labels, label_colors, legend=legend)
    plt.savefig(os.path.join(output_folder, 'tsne.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, 'tsne.svg'), bbox_inches='tight')

    # t-SNE Plot of some random (raw) embeddings
    plt.clf()
    embeddings_raw = np.concatenate((embeddings_8, embeddings_2))  # original embeddings
    map_labels_raw = np.concatenate((map_labels_8, map_labels_2))

    embeddings_raw, map_labels_raw = shuffle(embeddings_raw, map_labels_raw)

    draw_t_sne(embeddings_raw[:50], labels, map_labels_raw[:50], label_colors, legend=legend)
    plt.savefig(os.path.join(output_folder, 'tsne_rand.png'), bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, 'tsne_rand.svg'), bbox_inches='tight')
    '''

    # Calculate MR
    mr, map_clusters, threshold = calculate_minimal_mr(embeddings_linkage, map_labels)
    legacy_mr = calculate_legacy_minimal_mr(embeddings_linkage, map_labels, config.getint(list, 'total_speakers'))

    # Save Matching of samples to clusters for the minimal MR
    with open(os.path.join(output_folder, 'cluster_map.txt'), mode='w+') as f:
        f.write('Matching of segments and clusters with minimal MR:\n')
        f.write('| {:8s} | {:8s} |\n'.format('Speaker', 'Cluster'))
        for label_no, cluster_no in zip(map_labels, map_clusters):
            f.write('| {:8s} | {:8d} |\n'.format(labels[label_no], cluster_no))

    # Print results
    logger.info('MR = {:.4f} \t Threshold = {:.4f} \t (Legacy MR = {:.4f})'.format(mr, threshold, legacy_mr))


if __name__ == '__main__':

    config_path = './config/022_100_densefactor.cfg'

    for listNumber in [1, 2, 3]:
        print('List ' + listNumber.__str__())
        for checkpoint in list(range(999, 30999, 1000)):
            print('Checkpoint ' + checkpoint.__str__())
            test(checkpoint, listNumber, config_path, nchw=False)
