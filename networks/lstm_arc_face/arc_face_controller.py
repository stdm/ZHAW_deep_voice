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
        self.batch_size = 32


    def train_network(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

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

        net = ArcFaceBlock(num_speakers)
        net.hybridize()
        net.initialize(mx.init.Xavier())
        net.collect_params().reset_ctx(ctx)

        #kv = mx.kv.create('device')
        kv = mx.kv.create('local')
        trainer = mx.gluon.Trainer(net.collect_params(), mx.optimizer.AdaDelta(), kvstore=kv)

        loss = mx.ndarray.SoftmaxOutput
        metric = mx.metric.CompositeEvalMetric([AccMetric()])

        #loss = gluon.loss.SoftmaxCrossEntropyLoss(weight = 1.0)
        #loss = gluon.loss.SoftmaxCrossEntropyLoss()
        num_epochs = 0
        while num_epochs < 10:
            #trainer = update_learning_rate(opt.lr, trainer, epoch, opt.lr_factor, lr_steps)
            tic = time.time()
            train_iter.reset()
            metric.reset()
            btic = time.time()
            for i, batch in enumerate(train_iter):
                data = mx.gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
                label = mx.gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
                outputs = []
                Ls = []
                with mx.autograd.record():
                    for x, y in zip(data, label):
                        z = net(x, y)
                        L = loss(z, y)
                        #L = L/args.per_batch_size
                        Ls.append(L)
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
                if i>0 and i%20==0:
                    name, acc = metric.get()
                    if len(name)==2:
                        logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f, %s=%f'%(
                                     num_epochs, i, self.batch_size/(time.time()-btic), name[0], acc[0], name[1], acc[1]))
                    else:
                        logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f'%(
                                     num_epochs, i, self.batch_size/(time.time()-btic), name[0], acc[0]))
                    #metric.reset()
                btic = time.time()

            epoch_time = time.time()-tic

            # First epoch will usually be much slower than the subsequent epics,
            # so don't factor into the average
            if num_epochs > 0:
                total_time = total_time + epoch_time

            #name, acc = metric.get()
            #logger.info('[Epoch %d] training: %s=%f, %s=%f'%(num_epochs, name[0], acc[0], name[1], acc[1]))
            logger.info('[Epoch %d] time cost: %f'%(num_epochs, epoch_time))
            num_epochs = num_epochs + 1
            #name, val_acc = test(ctx, val_data)
            #logger.info('[Epoch %d] validation: %s=%f, %s=%f'%(epoch, name[0], val_acc[0], name[1], val_acc[1]))

            # save model if meet requirements
            #save_checkpoint(epoch, val_acc[0], best_acc)
        if num_epochs > 1:
            print('Average epoch time: {}'.format(float(total_time)/(num_epochs - 1)))
