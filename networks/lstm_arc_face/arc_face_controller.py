import numpy as np
import mxnet as mx
import time
import math
import os

from mxnet.gluon import nn
from mxnet.gluon import rnn
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric, check_label_shapes

from .data_generator import load_data
from .model import ArcFaceBlock
from .metrics import CrossEntropy
from .executor import run_epoch
from .saver import save_epoch

from common.utils.paths import *
from common.network_controller import NetworkController
from networks.lstm_arc_face import settings


class ArcFaceController(NetworkController):
    def __init__(self, train_data_name, val_data_name):
        super().__init__("arc_face", val_data_name)
        self.train_data_name = train_data_name
        self.network_file = self.name + '/' + self.train_data_name
        self.train_data_path = get_speaker_pickle(self.train_data_name)
        self.max_epochs = settings.MAX_EPOCHS
        self.batch_size = settings.BATCH_SIZE
        self.batches_per_epoch = settings.BATCHES_PER_EPOCH

    def train_network(self):
        ctx = []
        cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
        if len(cvd) > 0:
            for i in range(len(cvd.split(','))):
                ctx.append(mx.gpu(i))
        if len(ctx) == 0:
            ctx = [mx.cpu()]

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
        with open('accs.csv', 'w+') as file:
            file.write('epoch, train_acc, val_acc, train_loss, val_loss\n')
        while epoch < self.max_epochs:
            name, indices, mean_loss, time_used = run_epoch(net, ctx, train_iter, metric, trainer, loss, epoch, train=True)
            best_values = save_epoch(net, self.network_file, epoch, best_values, name, indices, mean_loss, time_used, save_rules, train=True)

            name, indices, mean_loss, time_used = run_epoch(net, ctx, val_iter, metric, trainer, loss, epoch, train=False)
            best_values = save_epoch(net, self.network_file, epoch, best_values, name, indices, mean_loss, time_used, save_rules, train=False)

            epoch = epoch + 1

        net.save_parameters('final_epoch')
