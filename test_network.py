# %%
import mxnet as mx

data = mx.sym.var('data')
state1 = mx.sym.var('state1', init=mx.init.Normal(0.01))
statecell1 = mx.sym.var('statecell1', init=mx.init.Normal(0.01))
param1 = mx.sym.var('param1', init=mx.init.Normal(0.01))
lstm1 = mx.sym.RNN(data=data, parameters=param1, state=state1, state_cell=statecell1, mode='lstm', state_size=256, num_layers=1)
drop1 = mx.sym.Dropout(data=lstm1, p=0.5)
state2 = mx.sym.var('state2', init=mx.init.Normal(0.01))
statecell2 = mx.sym.var('statecell2', init=mx.init.Normal(0.01))
param2 = mx.sym.var('param2', init=mx.init.Normal(0.01))
lstm2 = mx.sym.RNN(data=drop1, parameters=param2, state=state2, state_cell=statecell2, mode='lstm', state_size=256, num_layers=1)
dense1 = mx.sym.FullyConnected(data=lstm2, num_hidden=100 * 10)
drop2 = mx.sym.Dropout(data=dense1, p=0.25)
embedding = mx.sym.FullyConnected(data=drop2, num_hidden=100 * 5)
m = 0.5
# ArcFace Logits
all_label = mx.symbol.Variable('softmax_label')
gt_label = all_label
_weight = mx.symbol.Variable("last_fc_weight", shape=(100, 100 * 5), init=mx.init.Normal(0.01))
s = 64
_weight = mx.symbol.L2Normalization(_weight, mode='instance')
nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
last_fc = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=100, name='last fc')
s_m = s * m
gt_one_hot = mx.sym.one_hot(gt_label, depth = 100, on_value = s_m, off_value = 0.0)
last_fc = last_fc-gt_one_hot

out_list = [mx.symbol.BlockGrad(embedding)]
softmax = mx.symbol.SoftmaxOutput(data=last_fc, label = gt_label, name='softmax', normalization='valid')
out_list.append(softmax)
if True:
    #ce_loss = mx.symbol.softmax_cross_entropy(data=fc7, label = gt_label, name='ce_loss')/args.per_batch_size
    body = mx.symbol.SoftmaxActivation(data=last_fc)
    body = mx.symbol.log(body)
    _label = mx.sym.one_hot(gt_label, depth = 100, on_value = -1.0, off_value = 0.0)
    body = body*_label
    ce_loss = mx.symbol.sum(body)/32
    out_list.append(mx.symbol.BlockGrad(ce_loss))
model = mx.symbol.Group(out_list)

mx.viz.print_summary(model, shape={'data':(32, 128, 800), 'softmax_label':(32,)})
