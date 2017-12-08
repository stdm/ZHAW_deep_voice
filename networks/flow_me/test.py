#!/usr/bin/env python

import argparse
import os

import matplotlib
from sklearn.utils import shuffle

matplotlib.use('Agg')

import networks.flow_me.DataGen as DataGen
from common.analysis.old_cluster import *
from common.analysis.old_colors import *
from common.analysis.old_plots import *
from .nn.restore_session import *
from common.utils.load_config import *
from common.utils.logger import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Filter warnings out

# Read config
path_master_config = './config/master.cfg'

parser = argparse.ArgumentParser(description='Speaker Clustering with metric embedding. Clustering')
parser.add_argument('-c', dest='config', help='Path to config file', required=True)
parser.add_argument('-nchw', dest='nchw', action='store_true', help='NCHW Format, GPU only')
parser.add_argument('-ckpt', dest='ckpt', type=int, help='Desired Checkpoint', required=True)
parser.add_argument('-list', dest='list', type=int, default=1, help='Desired List to test. Default: List 1')

args = parser.parse_args()

cfg = load_config(path_master_config, args.config)

logger = get_logger('cluster', logging.INFO)

# Select List
if args.list == 1:
    list = 'test_list1'
elif args.list == 2:
    list = 'test_list2'
elif args.list == 3:
    list = 'test_list3'

output_layer_name = cfg.get('test', 'output_layer')

# Output folder
output_layer_name_folder = output_layer_name.replace('/', '-')
output_layer_name_folder = output_layer_name_folder.replace(':', '')

output_folder = os.path.join(cfg.get('output', 'plot_path'), cfg.get('exp', 'name'),
                             output_layer_name_folder, 'ckpt_{:05d}'.format(args.ckpt), cfg.get(list, 'name'))

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

add_file_handler(logger, os.path.join(output_folder, 'mr.log'))

# Load Session
path_meta_graph = os.path.join(cfg.get('output', 'log_path'), 'train_' + cfg.get('exp', 'name'),
                               'model.ckpt-' + str(args.ckpt) + '.meta')
path_ckpt = os.path.join(cfg.get('output', 'log_path'), 'train_' + cfg.get('exp', 'name'),
                         'model.ckpt-' + str(args.ckpt))

sess, input_layer, output_layer, train_mode = restore_session(path_meta_graph, path_ckpt, output_layer_name)

# Load test data
data_gen_8 = DataGen.DataGen(cfg, data_set=list, pickle_no='pickle1')
data_gen_2 = DataGen.DataGen(cfg, data_set=list, pickle_no='pickle2')

input_data_8, map_labels_8 = data_gen_8.get_timit_test_set(sentence_pickle=cfg.getint(list, 'sentences_pickle1'))
input_data_2, map_labels_2 = data_gen_2.get_timit_test_set(sentence_pickle=cfg.getint(list, 'sentences_pickle2'))

labels = data_gen_8.get_labels()  # Both data_gen instances should return the same labels

if not args.nchw:
    input_data_8 = np.transpose(input_data_8, axes=(0, 2, 3, 1))
    input_data_2 = np.transpose(input_data_2, axes=(0, 2, 3, 1))

label_colors = get_colors(len(labels))

# Pass samples trough network, get embeddings
embeddings_8 = sess.run(output_layer, feed_dict={input_layer: input_data_8, train_mode: False})
embeddings_2 = sess.run(output_layer, feed_dict={input_layer: input_data_2, train_mode: False})
sess.close()

# Reduce to mean, merge to one list of embeddings
# Init data structures according to used embeddings dimension
if 'l10_dense' in cfg.get('test', 'output_layer'):
    embeddings_mean_8 = np.zeros((cfg.getint(list, 'total_speakers'), int(cfg.getfloat(
        'net', 'dense10_factor') * cfg.getint('train', 'total_speakers'))))
    embeddings_mean_2 = np.zeros((cfg.getint(list, 'total_speakers'), int(cfg.getfloat(
        'net', 'dense10_factor') * cfg.getint('train', 'total_speakers'))))
elif 'l11_dense' in cfg.get('test', 'output_layer'):
    embeddings_mean_8 = np.zeros((cfg.getint(list, 'total_speakers'), int(cfg.getfloat(
        'net', 'dense11_factor') * cfg.getint('train', 'total_speakers'))))
    embeddings_mean_2 = np.zeros((cfg.getint(list, 'total_speakers'), int(cfg.getfloat(
        'net', 'dense11_factor') * cfg.getint('train', 'total_speakers'))))
else:
    embeddings_mean_8 = np.zeros((cfg.getint(list, 'total_speakers'), int(cfg.getfloat(
        'net', 'dense7_factor') * cfg.getint('train', 'total_speakers'))))
    embeddings_mean_2 = np.zeros((cfg.getint(list, 'total_speakers'), int(cfg.getfloat(
        'net', 'dense7_factor') * cfg.getint('train', 'total_speakers'))))

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
embeddings_dist, embeddings_linkage = cluster_embeddings(embeddings)

# Dendogram
if cfg.getint(list, 'total_speakers') > 10:
    draw_dendogram(embeddings_linkage, embeddings_dist, labels, map_labels, map_labels_full, label_colors,
                   figsize=(20, 4.8))
elif cfg.getint(list, 'total_speakers') > 40:
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
if cfg.getint(list, 'total_speakers') > 20:
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

# Calculate MR
mr, map_clusters, threshold = calculate_minimal_mr(embeddings_linkage, map_labels)
legacy_mr = calculate_legacy_minimal_mr(embeddings_linkage, map_labels, cfg.getint(list, 'total_speakers'))

# Save Matching of samples to clusters for the minimal MR
with open(os.path.join(output_folder, 'cluster_map.txt'), mode='w+') as f:
    f.write('Matching of segments and clusters with minimal MR:\n')
    f.write('| {:8s} | {:8s} |\n'.format('Speaker', 'Cluster'))
    for label_no, cluster_no in zip(map_labels, map_clusters):
        f.write('| {:8s} | {:8d} |\n'.format(labels[label_no], cluster_no))

# Print results
logger.info('MR = {:.4f} \t Threshold = {:.4f} \t (Legacy MR = {:.4f})'.format(mr, threshold, legacy_mr))
