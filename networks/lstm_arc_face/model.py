import mxnet as mx

import os

from mxnet.gluon import nn
from mxnet.gluon import rnn

def get_context():
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd) > 0:
        for i in range(len(cvd.split(','))):
            ctx.append(mx.gpu(i))
    if len(ctx) == 0:
        ctx = [mx.cpu()]
    return ctx

class NetworkBlock(mx.gluon.HybridBlock):
    def __init__(self, n_classes, settings, **kwargs):
        super(NetworkBlock, self).__init__(**kwargs)

        self.settings = settings
        self.lstm_hidden_1 = settings['LSTM_HIDDEN_1']
        self.lstm_hidden_2 = settings['LSTM_HIDDEN_2']
        self.dense_hidden_1 = n_classes * settings['DENSE_HIDDEN_1']
        self.dense_hidden_2 = n_classes * settings['DENSE_HIDDEN_2']
        self.drop_rate_1 = settings['DROP_RATE_1']
        self.drop_rate_2 = settings['DROP_RATE_2']

        self.output_size = self.dense_hidden_2

        with self.name_scope():
            self.embedding = nn.HybridSequential(prefix='')
            self.embedding.add(rnn.LSTM(self.lstm_hidden_1, bidirectional=True))
            self.embedding.add(nn.Dropout(self.drop_rate_1))
            self.embedding.add(rnn.LSTM(self.lstm_hidden_1, bidirectional=True))
            self.body = nn.HybridSequential(prefix='')
            self.body.add(nn.Dense(self.dense_hidden_1))
            self.body.add(nn.Dropout(self.drop_rate_2))
            self.body.add(nn.Dense(self.dense_hidden_2))

    def embeddings(self, x):
        x = self.embedding(x)
        if self.settings['EMBEDDINGS_FROM'] == 'LSTM':
            x = mx.ndarray.slice_axis(x, axis=1, begin=-1, end=None)
        elif self.settings['EMBEDDINGS_FROM'] == 'DENSE':
            x = self.body(x)
        return x

    def hybrid_forward(self, F, x):
        x = self.embedding(x)
        if self.settings['EMBEDDINGS_FROM'] == 'LSTM':
            x = F.slice_axis(x, axis=1, begin=-1, end=None)
        x = self.body(x)
        return x


class ArcFaceBlock(mx.gluon.HybridBlock):
    def __init__(self, n_classes, settings, **kwargs):
        super(ArcFaceBlock, self).__init__(**kwargs)
        self.s = settings['MARGIN_S']
        self.m1 = settings['MARGIN_M1']
        self.m2 = settings['MARGIN_M2']
        self.m3 = settings['MARGIN_M3']
        self.n_classes = n_classes
        with self.name_scope():
            self.body = nn.HybridSequential(prefix='')
            self.network_block = NetworkBlock(self.n_classes, settings)
            self.body.add(self.network_block)
            self.last_fc_weight = self.params.get('last_fc_weight', shape=(self.n_classes, self.network_block.output_size))

    def feature(self, x):
        return self.network_block.embeddings(x)

    def hybrid_forward(self, F, x, label, last_fc_weight):
        embeddings = self.body(x)

        norm_embeddings = F.L2Normalization(embeddings, mode='instance')
        norm_weights = F.L2Normalization(last_fc_weight, mode='instance')
        last_fc = F.FullyConnected(norm_embeddings, norm_weights, no_bias = True,
                                       num_hidden=self.n_classes, name='last_fc')

        original_target_logit = F.pick(last_fc, label, axis=1)
        theta = F.arccos(original_target_logit / self.s)
        if self.m1!=1.0:
            theta = theta*self.m1
        if self.m2>0.0:
            theta = theta+self.m2
        marginal_target_logit = F.cos(theta)
        if self.m3>0.0:
            marginal_target_logit = marginal_target_logit - self.m3
        gt_one_hot = F.one_hot(label, depth = self.n_classes, on_value = 1.0, off_value = 0.0)
        diff = marginal_target_logit * self.s
        diff = diff - original_target_logit
        diff = F.expand_dims(diff, 1)
        body = F.broadcast_mul(gt_one_hot, diff)
        out = last_fc + body
        return out, last_fc

class SoftmaxLoss(mx.gluon.HybridBlock):
    def __init__(self, n_classes, settings, **):
        self.n_classes = n_classes
        self.batch_size = settings['BATCH_SIZE']

    def hybrid_forward(self, F, x, label):
        body = F.SoftmaxActivation(x)
        body = F.log(body)
        gt_one_hot = F.one_hot(label, depth=self.n_classes, on_value=-1.0, off_value=0.0)
        body = body * gt_one_hot
        ce_loss = F.sum(body) / self.batch_size
        return ce_loss
