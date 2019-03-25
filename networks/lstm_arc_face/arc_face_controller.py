import mxnet as mx
import logging
import time
import os

from common.network_controller import NetworkController
from .data_generator import load_data
from .model import ArcFaceBlock
from .metrics import AccMetric
from common.utils.paths import *

class ArcFaceController(NetworkController):
    def __init__(self, train_data_name, val_data_name):
        super().__init__("arc_face", val_data_name)
        self.train_data_name = train_data_name
        self.network_file = self.name + self.train_data_name
        self.train_data_path = get_speaker_pickle(self.train_data_name)
        raw_max_epochs = input('Please enter the number of epochs:')
        raw_batch_size = input('Please enter the batch size you wish to use:')
        self.max_epochs = int(raw_max_epochs)
        self.batch_size = int(raw_batch_size)


    def train_network(self):
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

        train_iter, val_iter, num_speakers = load_data(self.train_data_path, self.batch_size)

        net = ArcFaceBlock(num_speakers, self.batch_size)
        net.hybridize()
        net.initialize(mx.init.Xavier())
        net.collect_params().reset_ctx(ctx)

        kv = mx.kv.create('device')
        #kv = mx.kv.create('local')
        trainer = mx.gluon.Trainer(net.collect_params(), mx.optimizer.AdaDelta(), kvstore=kv)

        metric = mx.metric.CompositeEvalMetric([AccMetric()])

        #loss = mx.ndarray.SoftmaxOutput
        num_epochs = 0
        total_time = 0
        lowest_train_loss = 100000
        lowest_val_loss = 100000
        while num_epochs < self.max_epochs:
            #trainer = update_learning_rate(opt.lr, trainer, epoch, opt.lr_factor, lr_steps)
            tic = time.time()
            train_iter.reset()
            val_iter.reset()
            metric.reset()
            btic = time.time()
            for i, batch in enumerate(train_iter):
                data = mx.gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
                label = mx.gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
                outputs = []
                Ls = []
                with mx.autograd.record():
                    for x, y in zip(data, label):
                        z, L1, L2 = net(x, y)
                        #L = L/args.per_batch_size
                        Ls.append(L1)
                        outputs.append(z)
                        # store the loss and do backward after we have done forward
                        # on all GPUs for better speed on multiple GPUs.
                    mx.autograd.backward(Ls)
                #trainer.step(batch.data[0].shape[0], ignore_stale_grad=True)
                #trainer.step(args.ctx_num)
                n = batch.data[0].shape[0]
                #print(n,n)
                trainer.step(n)
                metric.update(label, outputs)

                mean_loss = 0.0
                for L in Ls:
                    mean_loss += L.asnumpy().mean() / float(len(Ls))
                if mean_loss < lowest_train_loss:
                    lowest_train_loss = mean_loss
                btic = time.time()

            epoch_time = time.time()-tic
            name, train_acc = metric.get()
            metric.reset()

            for i, batch in enumerate(val_iter):
                data = mx.gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
                label = mx.gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
                outputs = []
                Ls = []
                for x, y in zip(data, label):
                    z, L1, L2 = net(x, y)
                    Ls.append(L1)
                    outputs.append(z)
                metric.update(label, outputs)

                mean_loss = 0.0
                for L in Ls:
                    mean_loss += L.asnumpy().mean() / float(len(Ls))
                if mean_loss < lowest_val_loss:
                    lowest_val_loss = mean_loss

            if num_epochs > 0:
                total_time = total_time + epoch_time

            name, val_acc = metric.get()
            print('[Epoch %d]\t time cost: %f\ttrain: %s=%f\tL=%f\tval: %s=%f\tL=%f'%(
                  num_epochs, epoch_time, name[0], train_acc[0], lowest_train_loss, name[0], val_acc[0], lowest_val_loss))
            num_epochs = num_epochs + 1
            #name, val_acc = test(ctx, val_data)
            #logger.info('[Epoch %d] validation: %s=%f, %s=%f'%(epoch, name[0], val_acc[0], name[1], val_acc[1]))

            # save model if meet requirements
            #save_checkpoint(epoch, val_acc[0], best_acc)
        if num_epochs > 1:
            print('Average epoch time: {}'.format(float(total_time)/(num_epochs - 1)))
