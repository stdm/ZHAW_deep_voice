import numpy as np
import mxnet as mx
import time
import math
import os

from tqdm import tqdm
from mxnet.gluon import nn
from mxnet.gluon import rnn
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric, check_label_shapes

from .data_generator import load_train_data, load_test_data
from .model import ArcFaceBlock, get_context, SoftmaxLoss
from .metrics import CrossEntropy
from .executor import run_epoch
from .saver import save_epoch, reset_progress, save_settings, plot_progress
from .loader import get_untrained_settings, get_trained_settings, get_params, get_trainer_params, get_last_epoch, extend_most_trained

from common.utils.paths import *
from common.network_controller import NetworkController
from common.clustering.generate_embeddings import generate_embeddings
from common.clustering.generate_clusters import cluster_embeddings
from common.analysis.analysis import *


class ArcFaceController(NetworkController):
    def __init__(self):
        super().__init__("arc_face", 'empty')

    def train_network(self):
        for settings in get_untrained_settings():
            save_settings(settings)
            epoch, _ = get_last_epoch(settings)
            if epoch == -1:
                extend_most_trained(settings)
                epoch, _ = get_last_epoch(settings)

            path = settings['SAVE_PATH']
            for d in path.split('/'):
                for p in d.split(';'):
                    print(p)
                print('')

            ctx = get_context()
            metric = CompositeEvalMetric([Accuracy(), TopKAccuracy(5), CrossEntropy()])
            save_rules = ['+', 'n', 'n']

            train_iter, val_iter, num_speakers = load_train_data(settings)

            net = ArcFaceBlock(num_speakers, settings)
            net.hybridize()
            trainer = None

            kv = mx.kv.create('device')

            if epoch == -1:
                net.initialize(ctx=ctx)
                net.collect_params().reset_ctx(ctx)
                trainer = mx.gluon.Trainer(net.collect_params(), mx.optimizer.AdaDelta(), kvstore=kv)
                epoch = 0
            else:
                reset_progress(settings)
                net.load_parameters(get_params(settings), ctx=ctx)
                trainer = mx.gluon.Trainer(net.collect_params(), mx.optimizer.AdaDelta(), kvstore=kv)
                trainer.load_states(get_trainer_params(settings))

            if settings['SOFTMAX'] == 'GLUON':
                loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
            elif settings['SOFTMAX'] == 'CUSTOM':
                loss = SoftmaxLoss(num_speakers, settings)
            best_values = {}
            while epoch < settings['MAX_EPOCHS']:
                name, indices, mean_loss, time_used = run_epoch(net, ctx, train_iter, metric, trainer, loss, epoch, train=True)
                best_values = save_epoch(net, trainer, settings, epoch, best_values, name, indices, mean_loss, time_used, save_rules, train=True)

                name, indices, mean_loss, time_used = run_epoch(net, ctx, val_iter, metric, trainer, loss, epoch, train=False)
                best_values = save_epoch(net, trainer, settings, epoch, best_values, name, indices, mean_loss, time_used, save_rules, train=False)
                print('')
                if epoch%10==0:
                    plot_progress(settings)
                if epoch%1000==0:
                    self.test_settings(settings)
                epoch = epoch + 1

            self.test_settings(settings)

    def test_settings(self, settings):
        plot_progress(settings)
        checkpoint_names, set_of_embeddings, set_of_true_clusters, embeddings_numbers = self.get_embeddings(settings)
        set_of_predicted_clusters = cluster_embeddings(set_of_embeddings)

        set_of_mrs = []
        set_of_homogeneity_scores = []
        set_of_completeness_scores = []

        for index, predicted_clusters in enumerate(set_of_predicted_clusters):
            mrs, homogeneity_scores, completeness_scores = calculate_analysis_values(predicted_clusters, set_of_true_clusters[index])

            set_of_mrs.append(mrs)
            set_of_homogeneity_scores.append(homogeneity_scores)
            set_of_completeness_scores.append(completeness_scores)

        fig = plot_curves('temp', [settings['NAME']], set_of_mrs, set_of_homogeneity_scores, set_of_completeness_scores, embeddings_numbers)
        fig.savefig(get_experiment_nets(settings['SAVE_PATH'] + '/clustering_results'))
        fig.savefig(get_experiment_nets(settings['SAVE_PATH'] + '/clustering_results.svg'), format='svg')

    def test_network(self, out_layer, seg_size, vec_size):
        for settings in get_trained_settings():
            self.test_settings(settings)

    def get_num_batches(self, data, settings):
        r = (len(data) - len(data)%settings['BATCH_SIZE'])/settings['BATCH_SIZE']
        r += 1 if len(data)%settings['BATCH_SIZE'] > 0 else 0
        return r

    def get_embeddings(self, settings):
        _, _, num_speakers = load_train_data(settings)

        net = ArcFaceBlock(num_speakers, settings)
        net.hybridize()
        checkpoints = [get_params(settings)]
        net.load_parameters(checkpoints[0])

        ctx = get_context()

        # Load and prepare train/test data
        x_train, speakers_train, x_test, speakers_test = load_test_data(settings)

        test_output, train_output = None, None

        start = 0
        check = True
        with tqdm(total=self.get_num_batches(x_test, settings), desc='getting test features') as pbar:
            while check:
                if start+settings['BATCH_SIZE'] >= len(x_test) - 1:
                    samples = mx.nd.array(x_test[start:-1])
                    check = False
                else:
                    samples = mx.nd.array(x_test[start:start+settings['BATCH_SIZE']])
                if start == 0:
                    test_output = net.feature(samples).asnumpy()
                else:
                    output = net.feature(samples).asnumpy()
                    test_output = np.concatenate((test_output, output))
                start += settings['BATCH_SIZE']
                pbar.update()
        start = 0
        check = True
        with tqdm(total=self.get_num_batches(x_train, settings), desc='getting train features') as pbar:
            while check:
                if start+settings['BATCH_SIZE'] >= len(x_train) - 1:
                    samples = mx.nd.array(x_train[start:-1])
                    check = False
                else:
                    samples = mx.nd.array(x_train[start:start+settings['BATCH_SIZE']])
                if start == 0:
                    train_output = net.feature(samples).asnumpy()
                else:
                    output = net.feature(samples).asnumpy()
                    train_output = np.concatenate((train_output, output))
                start += settings['BATCH_SIZE']
                pbar.update()


        test_output = np.squeeze(np.array(test_output))
        train_output = np.squeeze(np.array(train_output))

        vector_size = train_output.shape[1:]

        print('test_output len -> ' + str(test_output.shape))
        print('train_output len -> ' + str(train_output.shape))

        embeddings, speakers, num_embeddings = generate_embeddings(train_output, test_output, speakers_train, speakers_test, vector_size)
        return checkpoints, [embeddings], [speakers], [num_embeddings]
