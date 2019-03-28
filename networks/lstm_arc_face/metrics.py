import mxnet as mx

class AccMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(AccMetric, self).__init__('acc', axis=self.axis, output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        self.count+=1
        #preds = [preds[1]] #use softmax output
        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
                pred_label = pred_label.asnumpy().astype('int32').flatten()
                label = label.asnumpy()
                if label.ndim==2:
                    label = label[:,0]
                label = label.astype('int32').flatten()
                assert label.shape==pred_label.shape
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)