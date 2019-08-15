
from keras.layers import Layer, Dense
from keras import backend as K
from common.utils.load_config import *
from common.utils.paths import *

import tensorflow as tf
import keras


# Constants
tf_l = tf.Variable(0., name='loss')
x = tf.constant(0.)
loss = tf.Variable(0.)
sum_loss = tf.Variable(0.)

def get_custom_objects(config):
    inst = AngularLoss(config)
    custom_objects = {'AngularLossDense': inst.get_dense(),
                      'angular_loss': inst.angular_loss,
                      'pairwise_kl_divergence':pairwise_kl_divergence,
                      'orig_pairwise_kl_divergence':orig_pairwise_kl_divergence}
    return custom_objects

def get_loss(config):
    if config.get('train', 'loss') == 'angular_margin':
        return AngularLoss(config).angular_loss
    elif config.get('train', 'loss') == 'kldiv_orig':
        return orig_pairwise_kl_divergence
    return pairwise_kl_divergence

def add_final_layers(model, config):
    if config.get('train', 'loss') == 'angular_margin':
        inst = AngularLoss(config)
        model.add(inst.get_dense()())
    else:
        model.add(Dense(units=config.getint('train', 'n_speakers'), activation='softmax'))

# angular loss function
class AngularLoss:
    def __init__(self, config):
        self.n_speakers = config.getint('train', 'n_speakers')
        self.margin_cosface = config.getfloat('angular_loss', 'margin_cosface')
        self.margin_arcface = config.getfloat('angular_loss', 'margin_arcface')
        self.margin_sphereface = config.getfloat('angular_loss', 'margin_sphereface')
        self.scale = config.getfloat('angular_loss', 'scale')

    def get_dense(self):
        n_speakers = self.n_speakers
        class AngularLossDense(Layer):
            def __init__(self, **kwargs):
                super(AngularLossDense, self).__init__(**kwargs)

            def build(self, input_shape):
                super(AngularLossDense, self).build(input_shape[0])
                self.W = self.add_weight(name='W',
                                         shape=(input_shape[-1], n_speakers),
                                         initializer='glorot_uniform',
                                         trainable=True)

            def call(self, inputs):
                x = tf.nn.l2_normalize(inputs, axis=1)
                W = tf.nn.l2_normalize(self.W, axis=0)

                logits = x @ W
                return logits

            def compute_output_shape(self, input_shape):
                return (None, n_speakers)
        return AngularLossDense

    def angular_loss(self, y_true, y_pred):
        logits = y_pred
        if self.margin_sphereface != 1.0 or self.margin_arcface != 0.0:
            y_pred = K.clip(y_pred, -1.0 + K.epsilon(), 1.0 - K.epsilon())
            theta = tf.acos(y_pred)
            if self.margin_sphereface != 1.0:
                theta = theta * self.margin_sphereface
            if self.margin_arcface != 0.0:
                theta = theta + self.margin_arcface
            y_pred = tf.cos(theta)
        target_logits = y_pred
        if self.margin_cosface != 0:
            target_logits = target_logits - self.margin_cosface

        logits = logits * (1 - y_true) + target_logits * y_true
        logits *= self.scale

        out = tf.nn.softmax(logits)
        loss = keras.losses.categorical_crossentropy(y_true, out)
        return loss

# pairwise kldiv loss function
def pairwise_kl_divergence(labels, predictions):
    margin = tf.constant(3.)
    x = tf.constant(0)
    sum_loss = tf.while_loop(outerLoop_condition, outerLoop, [x, tf_l, predictions, labels, margin], swap_memory=True,
                             parallel_iterations=10, name='outerloop')
    n = tf.constant(100.)
    pairs = tf.multiply(n, tf.divide(tf.subtract(n, tf.constant(1.)), tf.constant(2.)))
    loss = tf.divide(sum_loss[1], pairs)
    return loss


def orig_pairwise_kl_divergence(labels, predictions):
    margin = tf.constant(2.)
    x = tf.constant(0)
    sum_loss = tf.while_loop(outerLoop_condition, outerLoop, [x, tf_l, predictions, labels, margin], swap_memory=True,
                             parallel_iterations=10, name='outerloop')
    n = tf.constant(100.)
    pairs = tf.multiply(n, tf.divide(tf.subtract(n, tf.constant(1.)), tf.constant(2.)))
    loss = tf.divide(sum_loss[1], pairs)
    return loss


def outerLoop_condition(x, tf_l, predictions, labels, margin):
    return tf.less(x, tf.constant(100))


def outerLoop(x, tf_l, predictions, labels, margin):
    def innerLoop(y, x, tf_l, predictions, labels, margin):
        tf_l = tf.add(tf_l, tf.cond(tf.greater(y, x),
                                    lambda: loss_with_kl_div(predictions[x], labels[x], predictions[y], labels[y],
                                                             margin), return_zero))
        y = tf.add(y, tf.constant(1))
        return y, x, tf_l, predictions, labels, margin

    def innerLoop_cond(y, x, tf_l, predictions, labels, margin):
        return tf.less(y, tf.constant(100))

    y = tf.constant(0)
    res = tf.while_loop(innerLoop_cond, innerLoop, [y, x, tf_l, predictions, labels, margin], swap_memory=True,
                        parallel_iterations=10, name='innerloop')
    return tf.add(x, 1), res[2], predictions, labels, margin


def loss_with_kl_div(P, xp, Q, xq, margin):
    epsilon = tf.constant(1e-16)
    P = tf.add(epsilon, P)
    Q = tf.add(epsilon, Q)

    Is = tf.cond(tf.reduce_all(tf.equal(xq, xp)), return_one, return_zero)
    Ids = tf.abs(tf.subtract(Is, tf.constant(1.)))

    KLPQ = tf.reduce_sum(tf.multiply(P, tf.log(tf.divide(P, Q))))
    KLQP = tf.reduce_sum(tf.multiply(Q, tf.log(tf.divide(Q, P))))
    lossPQ = tf.add(tf.multiply(Is, KLPQ), tf.multiply(Ids, tf.maximum(tf.constant(0.), tf.subtract(margin, KLPQ))))
    lossQP = tf.add(tf.multiply(Is, KLQP), tf.multiply(Ids, tf.maximum(tf.constant(0.), tf.subtract(margin, KLQP))))
    L = tf.add(lossPQ, lossQP)
    return L


def return_zero():
    return tf.constant(0.)


def return_one():
    return tf.constant(1.)
