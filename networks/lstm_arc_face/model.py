import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon import rnn

class NetworkBlock(mx.gluon.HybridBlock):
    def __init__(self, n_classes, **kwargs):
        super(NetworkBlock, self).__init__(**kwargs)

        self.lstm_hidden_1 = 256
        self.lstm_hidden_2 = 256
        self.dense_hidden_1 = n_classes * 10
        self.dense_hidden_2 = n_classes * 5
        self.drop_rate_1 = 0.5
        self.drop_rate_2 = 0.25

        self.output_size = self.dense_hidden_2

        with self.name_scope():
            self.body = nn.HybridSequential(prefix='')
            self.body.add(rnn.LSTM(self.lstm_hidden_1, bidirectional=True))
            self.body.add(nn.Dropout(self.drop_rate_1))
            self.body.add(rnn.LSTM(self.lstm_hidden_2, bidirectional=True))
            self.body.add(nn.Dense(self.dense_hidden_1))
            self.body.add(nn.Dropout(self.drop_rate_2))
            self.body.add(nn.Dense(self.dense_hidden_2))

    def feature(self, x):
        feat = self.body(x)
        return feat

    def hybrid_forward(self, F, x):
        x = self.body(x)
        return x


class ArcFaceBlock(mx.gluon.HybridBlock):
    def __init__(self, n_classes, **kwargs):
        super(ArcFaceBlock, self).__init__(**kwargs)
        self.s = 64
        self.m = 0.5
        self.n_classes = n_classes
        with self.name_scope():
            self.body = nn.HybridSequential(prefix='')

            network_block = NetworkBlock(self.n_classes)
            self.body.add(network_block)
            self.last_fc_weight = self.params.get('last_fc_weight', shape=(self.n_classes, network_block.output_size))

    def feature(self, x):
        feat = self.body(x)
        return feat

    def hybrid_forward(self, F, x, label, last_fc_weight):
        embeddings = x
        #embeddings = self.body(x)

        norm_embeddings = F.L2Normalization(embeddings, mode='instance')
        norm_weights = F.L2Normalization(last_fc_weight, mode='instance')
        last_fc = F.FullyConnected(norm_embeddings, norm_weights, no_bias = True,
                                   num_hidden=self.n_classes, name='last_fc')

        original_target_logit = F.pick(last_fc, label, axis=1)
        theta = F.arccos(original_target_logit)
        marginal_target_logit = F.cos(theta + self.m)
        gt_one_hot = F.one_hot(label, depth = self.n_classes, on_value = 1.0, off_value = 0.0)
        diff = marginal_target_logit - original_target_logit
        diff = F.expand_dims(diff, 1)
        body = F.broadcast_mul(gt_one_hot, diff)
        last_fc = last_fc + body
        last_fc = last_fc * self.s
        return last_fc
