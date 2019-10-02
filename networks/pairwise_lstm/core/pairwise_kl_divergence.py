"""
    Providing the loss function for the LSTM Keras model, pairwise_kl_divergence is implemented in tensorflow.

    Function layout in this file has been altered slightly from given files.
    Work of Gerber and Glinski.
"""
import tensorflow as tf

tf_l = tf.Variable(0., name='loss')
x = tf.constant(0.)
loss = tf.Variable(0.)
sum_loss = tf.Variable(0.)


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


if __name__ == "__main__":

    #Initialize test environment
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    #Test 1
    #Initialize functions under test
    predictions = tf.placeholder('float', [None, None])
    targets = tf.placeholder('float', None)
    lstm_loss = pairwise_kl_divergence(targets, predictions)
    kldiv_loss = orig_pairwise_kl_divergence(targets, predictions)

    #Prepare test data
    nr_of_elems = 100
    test_pred = [None] * nr_of_elems
    for i in range(nr_of_elems):
        test_pred[i] = [1, 2, 3]
    test_pred[50] = [2, 3, 4]
    test_targ = [None] * nr_of_elems
    for i in range(nr_of_elems):
        test_targ[i] = 1

    #Run tests
    res = sess.run(lstm_loss, feed_dict={targets: test_targ, predictions: test_pred})
    print(res)

    res = sess.run(kldiv_loss, feed_dict={targets: test_targ, predictions: test_pred})
    print(res)

    #Test 2
    # Initialize functions under test
    P = tf.placeholder('float', None)
    xp = tf.placeholder('float', None)
    Q = tf.placeholder('float', None)
    xq = tf.placeholder('float', None)
    margin = tf.placeholder('float', None)
    loss_kl_div = loss_with_kl_div(P, xp, Q, xq, margin)

    #Prepare test data
    t_P = [1,2,3]
    t_xp = 1
    t_Q = [2,3,4]
    t_xq = 1
    t_margin = 2

    #Run tests
    res = sess.run(loss_kl_div, feed_dict={P: t_P, xp: t_xp, Q: t_Q, xq: t_xq, margin: t_margin})
    print(res)
