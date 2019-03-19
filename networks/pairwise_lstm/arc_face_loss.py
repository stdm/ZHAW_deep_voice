from keras import backend as K
import tensorflow as tf
from keras.layers import Layer

n_classes = -1

class ArcFace(Layer):

    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        embds = K.l2_normalize(inputs, axis=1, name='normed_embd')
        weights = K.l2_normalize(self.kernel, axis=0)

        output = K.dot(embds, weights, name='cos_t')
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def calculate_loss(labels, predictions):
        m = 0.5
        s = 64.0

        cos_m = math.cos(m)
        sin_m = math.sin(m)

        mm = sin_m * m

        threshold = math.cos(math.pi - m)

        cos_t2 = tf.square(predictions, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')

        sin_t = tf.sqrt(sin_t2, name='sin_t')

        cos_mt = s * tf.subtract(tf.multiply(predictions, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
        cond_v = predictions - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
        keep_val = s*(predictions - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)
        mask = tf.one_hot(labels, depth=self.units, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')
        s_cos_t = tf.multiply(s, predictions, name='scalar_cos_t')
        logits = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')

        inference_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
        wd_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        train_loss = inference_loss + wd_loss
        print(train_loss)
        return train_loss
    #TODO add softmax
