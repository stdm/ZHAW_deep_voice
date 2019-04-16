import mxnet as mx
import numpy as np

class CrossEntropy(mx.metric.EvalMetric):
    def __init__(self, eps=1e-12, name='cross-entropy', output_names=None, label_names=None):
        super(CrossEntropy, self).__init__(name, eps=eps,output_names=output_names, label_names=label_names)
        self.eps = eps

    def update(self, labels, preds):
        labels, preds = mx.metric.check_label_shapes(labels, preds, True)
        for label, pred in zip(labels, preds):
            pred = mx.nd.softmax(pred, axis=1)
            label = label.asnumpy()
            pred = pred.asnumpy()

            label = label.ravel()
            assert label.shape[0] == pred.shape[0]
            prob = pred[np.arange(label.shape[0]), np.int64(label)]
            self.sum_metric += (-np.log(prob + self.eps)).sum()
            self.num_inst += label.shape[0]
