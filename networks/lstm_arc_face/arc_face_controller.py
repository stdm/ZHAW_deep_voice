import numpy as np
import mxnet as mx
import time
import math
import os

from mxnet.gluon import nn
from mxnet.gluon import rnn
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric, check_label_shapes

from .data_generator import load_data, load_test_data
from .model import ArcFaceBlock, get_context
from .metrics import CrossEntropy
from .executor import run_epoch
from .saver import save_epoch, reset_progress, save_final, get_params

from common.utils.paths import *
from common.network_controller import NetworkController
from common.clustering.generate_embeddings import generate_embeddings
from networks.lstm_arc_face import settings


class ArcFaceController(NetworkController):
    def __init__(self, train_data_name=settings.TRAIN_DATA_NAME, val_data_name=settings.VAL_DATA_NAME):
        super().__init__("arc_face", val_data_name)
        self.train_data_name = train_data_name
        self.network_file = self.name + '/' + self.train_data_name
        self.train_data_path = get_speaker_pickle(self.train_data_name)
        self.max_epochs = settings.MAX_EPOCHS
        self.batch_size = settings.BATCH_SIZE
        self.batches_per_epoch = settings.BATCHES_PER_EPOCH

    def train_network(self):
        reset_progress(self.network_file)
        ctx = get_context()
        metric = CompositeEvalMetric([Accuracy(), TopKAccuracy(5), CrossEntropy()])
        save_rules = ['+', 'n', 'n']

        train_iter, val_iter, num_speakers = load_data(self.train_data_path, self.batch_size, self.batches_per_epoch)

        net = ArcFaceBlock(num_speakers)
        net.hybridize()
        net.initialize(mx.init.Xavier())
        net.collect_params().reset_ctx(ctx)

        kv = mx.kv.create('device')
        trainer = mx.gluon.Trainer(net.collect_params(), mx.optimizer.AdaDelta(), kvstore=kv)

        loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

        epoch = 0
        best_values = {}
        while epoch < self.max_epochs:
            name, indices, mean_loss, time_used = run_epoch(net, ctx, train_iter, metric, trainer, loss, epoch, train=True)
            best_values = save_epoch(net, self.network_file, epoch, best_values, name, indices, mean_loss, time_used, save_rules, train=True)

            name, indices, mean_loss, time_used = run_epoch(net, ctx, val_iter, metric, trainer, loss, epoch, train=False)
            best_values = save_epoch(net, self.network_file, epoch, best_values, name, indices, mean_loss, time_used, save_rules, train=False)
            print('')
            epoch = epoch + 1

        save_final(net, self.network_file)

    def get_embeddings(self, out_layer, seg_size, vec_size):
        _, _, num_speakers = load_data(self.train_data_path, self.batch_size, self.batches_per_epoch)

        net = ArcFaceBlock(num_speakers)
        net.hybridize()
        checkpoints = [get_params(self.network_file)]
        net.load_parameters(checkpoints[0])

        ctx = get_context()

        # Load and prepare train/test data
        x_test, speakers_test = load_test_data(self.get_validation_test_data(), seg_size)
        x_train, speakers_train = load_test_data(self.get_validation_train_data(), seg_size)

        x_test = mx.nd.array(x_test)
        x_train = mx.nd.array(x_train)

        # Prepare return values
        set_of_embeddings = []
        set_of_speakers = []
        speaker_numbers = []

        test_output = net.feature(x_test).asnumpy()
        train_output = net.feature(x_train).asnumpy()

        vector_size = train_output.shape[-1]

        print('test_output len -> ' + str(test_output.shape))
        print('train_output len -> ' + str(train_output.shape))

        embeddings, speakers, num_embeddings = generate_embeddings(train_output, test_output, speakers_train,
                                                                       speakers_test, vector_size)

        # Fill the embeddings and speakers into the arrays
        set_of_embeddings.append(embeddings)
        set_of_speakers.append(speakers)
        speaker_numbers.append(num_embeddings)

        print('Pairwise_lstm test done.')
        return checkpoints, set_of_embeddings, set_of_speakers, speaker_numbers
