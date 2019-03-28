import mxnet as mx
import numpy as np
import time

def test_batch(net, ctx, batch, metric, loss):
    data = mx.gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
    label = mx.gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
    outputs, losses = [], []
    for x, y in zip(data, label):
        z, ze = net(x, y)
        losses.append(loss(z, y).asnumpy())
        outputs.append(ze)
    metric.update(label, outputs)
    return np.mean(np.array(losses))

def train_batch(net, ctx, batch, metric, loss):
    data = mx.gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
    label = mx.gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
    outputs, losses, Ls = [], [], []
    with mx.autograd.record():
        for x, y in zip(data, label):
            z, ze = net(x, y)
            Ls.append(loss(z, y))
            losses.append(Ls[-1].asnumpy())
            outputs.append(ze)
        mx.autograd.backward(Ls)
    metric.update(label, outputs)
    return np.mean(np.array(losses))

def run_epoch(net, ctx, data, metric, trainer, loss, epoch, train=True, verbose=True):
    data.reset()
    metric.reset()
    losses = []
    start = time.time()
    for batch in data:
        if train:
            losses.append(train_batch(net, ctx, batch, metric, loss))
            trainer.step(batch.data[0].shape[0])
        else:
            losses.append(test_batch(net, ctx, batch, metric, loss))
    time_used = time.time() - start
    name, indices = metric.get()
    if verbose:
        text = '[Epoch %d]\t'%epoch
        text += 'training:\t' if train else 'validating:\t'
        text += 'time=%.6f'(time_used)
        for i in range(len(name)):
            text += '\t%s=%.6f'%(name[i],indices[i])
        print(text)
    return name, indices, np.mean(np.array(losses)), time_used
