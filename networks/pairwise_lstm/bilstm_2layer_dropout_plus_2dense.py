import pickle

import numpy as np

import common.spectogram.speaker_train_splitter as sts
from .core import plot_saver as ps

np.random.seed(1337)  # for reproducibility
import mxnet as mx
import numpy as np

from random import randint
import sys
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import CuDNNLSTM
from keras.layers.wrappers import Bidirectional
from .core import data_gen as dg
from .core import pairwise_kl_divergence as kld
from .arc_face_loss import ArcFace
from .metric import *
from .custom_iterator import CustomIterator
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))

from common.utils.paths import *

'''This Class Trains a Bidirectional LSTM with 2 Layers, and 2 Denselayer and a Dropout Layers
    Parameters:
    name: Network/experiment name to save artifacts
    training_data: name of the Training data file, expected to be train_xxxx.pickle
    n_hidden1: Units of the first LSTM Layer
    n_hidden2: Units of the second LSTM Layer
    n_classes: Amount of output classes (Speakers in Trainingset)
    n_10_batches: Number of Minibatches to train the Network (1 = 10 Minibatches)
    segment_size: Segment size that is used as input 100 equals 1 second with current Spectrogram extraction
    frequency: size of the frequency Dimension of the Input Spectrogram

    Work of Gerber and Glinski.
'''


class bilstm_2layer_dropout(object):
    def __init__(self, name, training_data, n_hidden1, n_hidden2, n_classes, n_10_batches,
                 segment_size, m, s, frequency=128):
        self.ce_loss = True
        self.batch_size = 64
        self.per_batch_size = 32
        self.network_name = name
        self.training_data = training_data
        self.test_data = 'test' + training_data[5:]
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_classes = n_classes
        self.n_10_batches = n_10_batches
        self.segment_size = segment_size
        self.input = (segment_size, frequency)
        self.m = m
        self.s = s
        self.run_network()

    def create_net(self):
        data = mx.sym.var('data')
        lstm1 = mx.sym.RNN(data=data, mode='lstm', state_size=self.n_hidden1, num_layers=1)
        drop1 = mx.sym.Dropout(data=lstm1, p=0.5)
        lstm2 = mx.sym.RNN(data=drop1, mode='lstm', state_size=self.n_hidden2, num_layers=1)
        dense1 = mx.sym.FullyConnected(data=lstm2, num_hidden=self.n_classes * 10)
        drop2 = mx.sym.Dropout(data=dense1, p=0.25)
        embedding = mx.sym.FullyConnected(data=drop2, num_hidden=self.n_classes * 5)

        # ArcFace Logits
        all_label = mx.symbol.Variable('softmax_label')
        gt_label = all_label
        _weight = mx.symbol.Variable("last_fc_weight", shape=(self.n_classes, self.n_classes * 5), init=mx.init.Normal(0.01))
        s = self.s
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
        last_fc = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=self.n_classes, name='last fc')
        s_m = s * self.m
        gt_one_hot = mx.sym.one_hot(gt_label, depth = self.n_classes, on_value = s_m, off_value = 0.0)
        last_fc = last_fc-gt_one_hot

        out_list = [mx.symbol.BlockGrad(embedding)]
        softmax = mx.symbol.SoftmaxOutput(data=last_fc, label = gt_label, name='softmax', normalization='valid')
        out_list.append(softmax)
        if self.ce_loss:
            #ce_loss = mx.symbol.softmax_cross_entropy(data=fc7, label = gt_label, name='ce_loss')/args.per_batch_size
            body = mx.symbol.SoftmaxActivation(data=last_fc)
            body = mx.symbol.log(body)
            _label = mx.sym.one_hot(gt_label, depth = self.n_classes, on_value = -1.0, off_value = 0.0)
            body = body*_label
            ce_loss = mx.symbol.sum(body)/self.per_batch_size
            out_list.append(mx.symbol.BlockGrad(ce_loss))
        model = mx.symbol.Group(out_list)
        print()
        print()
        print()
        mx.viz.print_summary(model, shape={'data':(32, 128, 800), 'softmax_label':(32, 100)})
        input('test')
        return model

    def prep_iter(self, X, labels):
        speakers = np.amax(labels) + 1
        Y = []
        for label in labels:
            y = np.zeros(speakers)
            y[label] = 1
            Y.append(y)
        return mx.io.NDArrayIter(data=np.squeeze(X), label=np.array(Y), batch_size=self.batch_size)

    def create_train_data(self):
        with open(get_speaker_pickle(self.training_data), 'rb') as f:
            (X, y, speaker_names) = pickle.load(f)

        splitter = sts.SpeakerTrainSplit(0.2, 10)
        X_t, X_v, y_t, y_v = splitter(X, y)

        train_iter = self.prep_iter(X_t, y_t)
        test_iter = self.prep_iter(X_v, y_v)
        return train_iter, test_iter

    def run_network(self):
        sym = self.create_net()
        ctx = []
        cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
        if len(cvd)>0:
            for i in range(len(cvd.split(','))):
                ctx.append(mx.gpu(i))
        if len(ctx)==0:
            ctx = [mx.cpu()]
            print('use cpu')
        else:
            print('gpu num:', len(ctx))

        model = mx.mod.Module(
            context       = ctx,
            symbol        = sym,
        )
        metric1 = AccMetric()
        eval_metrics = [mx.metric.create(metric1)]
        if self.ce_loss:
            metric2 = LossValueMetric()
            eval_metrics.append( mx.metric.create(metric2) )

        train_iter, test_iter = self.create_train_data()

        initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)

        opt = mx.optimizer.AdaDelta()

        highest_acc = [0.0, 0.0]  #lfw and target
        #for i in range(len(ver_list)):
        #  highest_acc.append(0.0)
        global_step = [0]
        save_step = [0]


        prefix = os.path.join(get_experiment_nets(self.network_name), 'test', 'model')
        prefix_dir = os.path.dirname(prefix)
        print('prefix', prefix)
        if not os.path.exists(prefix_dir):
            os.makedirs(prefix_dir)

        def ver_test(nbatch):
            results = []
            for i in range(len(ver_list)):
                acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model, self.batch_size, 10, None, None)
                print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
                #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
                print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
                results.append(acc2)
            return results

        def _batch_callback(param):
            global_step[0]+=1
            mbatch = global_step[0]
            _cb(param)
            if mbatch%100==0:
                print('lr-batch-epoch:', param.nbatch, param.epoch)

            if mbatch>=0:
                acc_list = ver_test(mbatch)
                save_step[0]+=1
                msave = save_step[0]
                do_save = False
                is_highest = False
                if len(acc_list)>0:
                    #lfw_score = acc_list[0]
                    #if lfw_score>highest_acc[0]:
                    #  highest_acc[0] = lfw_score
                    #  if lfw_score>=0.998:
                    #    do_save = True
                    score = sum(acc_list)
                    if acc_list[-1]>=highest_acc[-1]:
                        if acc_list[-1]>highest_acc[-1]:
                            is_highest = True
                        else:
                            if score>=highest_acc[0]:
                                is_highest = True
                                highest_acc[0] = score
                        highest_acc[-1] = acc_list[-1]
                        #if lfw_score>=0.99:
                        #  do_save = True
                if is_highest:
                    do_save = True

                if do_save:
                    print('saving', msave)
                    arg, aux = model.get_params()
                    mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
                print('[%d]Accuracy-Highest: %1.5f'%(mbatch, highest_acc[-1]))

        model.fit(train_iter,
                  begin_epoch        = 0,
                  num_epoch          = 10,
                  eval_data          = test_iter,
                  eval_metric        = eval_metrics,
                  kvstore            = 'device',
                  optimizer          = opt,
                  #optimizer_params   = optimizer_params,
                  initializer        = initializer,
                  allow_missing      = True,
                  batch_end_callback = None)

        arg, aux = model.get_params()
        mx.model.save_checkpoint(prefix, global_step[0], model.symbol, arg, aux)
