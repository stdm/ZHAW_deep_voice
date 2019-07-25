"""
    This file provides the mean loss function with kl divergence.

    The exported function to the clustering networks is:
    - create_loss_functions_kl_div(input_var, network, target_var, margin)

    Work of Lukic and Vogt, adapted by Heusser
"""
import lasagne

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from common.utils.load_config import *
from common.utils.paths import *

epsilon = 1e-16  # to avoid log(0) or division by 0


def create_loss_functions_kl_div(input_var, network, target_var, margin):
    config = load_config(None, join(get_common(), 'config.cfg'))
    # define loss expression
    prediction = lasagne.layers.get_output(network)
    loss = mean_loss_kl_div(prediction, target_var, margin)

    # define update expression for training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(loss, params, config.getfloat('pairwise_kldiv', 'adadelta_learning_rate'), config.getfloat('pairwise_kldiv', 'adadelta_rho'), config.getfloat('pairwise_kldiv', 'adadelta_epsilon'))
    # updates = lasagne.updates.adagrad(loss, params, learning_rate=1.0, epsilon=1e-6)
    # updates = lasagne.updates.adam(loss, params, learning_rate=0.001, beta1=0.9, beta2 = 0.999, epsilon = 1e-8)
    # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.001)

    # loss expression for validation/testing (disable dropouts etc.)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = mean_loss_kl_div(test_prediction, target_var, margin)

    # create accuracy expression
    test_class_predict = T.argmax(test_prediction, axis=1)

    # compile theano functions for training and validation/accuracy
    train_fn = theano.function([input_var, target_var, margin], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var, margin], [test_loss, test_class_predict])  # , test_acc])
    return train_fn, val_fn


def mean_loss_kl_div(predictions, targets, margin):
    err_mat = foreach(predictions, targets, margin)
    length = err_mat.shape[0]
    mean_err = err_mat.sum() / ((length - 1) * length)
    return mean_err


def foreach(predictions, targets, margin):
    result, updates = theano.scan(fn=step,
                                  sequences=[predictions, targets],
                                  non_sequences=[predictions, targets, margin], strict=True)
    return result


def step(prediction, target, predictions, targets, margin):
    result, updates = theano.scan(fn=loss_with_kl_div,
                                  sequences=[predictions, targets],
                                  non_sequences=[prediction, target, margin], strict=True)
    return result


def loss_with_kl_div(P, xp, Q, xq, margin):
    P += epsilon
    Q += epsilon

    Is = ifelse(T.eq(xq, xp), 1., 0.)
    Ids = abs(Is - 1)

    KLPQ = T.sum(P * T.log(P / Q))
    KLQP = T.sum(Q * T.log(Q / P))
    lossPQ = Is * KLPQ + Ids * T.max([0, margin - KLPQ])
    lossQP = Is * KLQP + Ids * T.max([0, margin - KLQP])
    L = lossPQ + lossQP

    return L


if __name__ == "__main__":
    #Prepare test data
    nr_of_elems = 100
    test_pred = [None] * nr_of_elems
    for i in range(nr_of_elems):
        test_pred[i] = [1, 2, 3]
    test_pred[50] = [2, 3, 4]
    test_targ = [None] * nr_of_elems
    for i in range(nr_of_elems):
        test_targ[i] = 1

    #Prepare function under test
    test_margin = 2
    predictions = T.fmatrix('p')
    targets = T.dvector('t')
    margin = T.scalar('margin')
    loss = mean_loss_kl_div(predictions, targets, margin)
    loss_fun = theano.function([predictions, targets, margin], loss)

    #Run test
    mean_err = loss_fun(test_pred, test_targ, test_margin)
    print(mean_err)
