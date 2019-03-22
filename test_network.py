# %%
import mxnet as mx

data = mx.sym.var('data')

lstm_cell1 = mx.rnn.LSTMCell(num_hidden=256)
begin_state1 = lstm_cell1.begin_state()
lstm_output1, lstm_states1 = lstm_cell(data, begin_state1)

drop1 = mx.sym.Dropout(data=lstm_output1, p=0.5)

lstm_cell2 = mx.rnn.LSTMCell(num_hidden=256)
begin_state2 = lstm_cell2.begin_state()
lstm_output2, lstm_states2 = lstm_cell(data, begin_state2)

dense1 = mx.sym.FullyConnected(data=lstm_output2, num_hidden=100 * 10)
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
if False:
    #ce_loss = mx.symbol.softmax_cross_entropy(data=fc7, label = gt_label, name='ce_loss')/args.per_batch_size
    body = mx.symbol.SoftmaxActivation(data=last_fc)
    body = mx.symbol.log(body)
    _label = mx.sym.one_hot(gt_label, depth = 100, on_value = -1.0, off_value = 0.0)
    body = body*_label
    ce_loss = mx.symbol.sum(body)/32
    out_list.append(mx.symbol.BlockGrad(ce_loss))
model = mx.symbol.Group(out_list)

mx.viz.print_summary(model, shape={'data':(32, 128, 800), 'softmax_label':(32,)})
