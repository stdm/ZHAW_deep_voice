"""
    Providing the loss function for the LSTM Keras model, pairwise_kl_divergence is implemented in tensorflow.

    Function layout in this file has been altered slightly from given files.
    Work of Gerber and Glinski.
"""
import tensorflow as tf

tf_l = tf.Variable(0., name='loss')
x = tf.constant(0.)
margin = tf.constant(3.)
loss = tf.Variable(0.)
sum_loss = tf.Variable(0.)


def pairwise_kl_divergence(labels, predictions):
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
    epsilon = 1e-16
    test_pred = [[1., 2., 3.], [4., 2., 3.], [6., 3., 2.], [4., 1., 5.], [2., 5., 8.]]
    # test_targ = [1, 1, 6]
    test_targ = [[1., 0., 0.], [1., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 0.]]
    test_margin = 2.
    predictions = tf.placeholder('float', [None, None])
    targets = tf.placeholder('float', [None, None])
    margin = tf.placeholder('float', None)
    margin = tf.stack(test_margin)
    result = pairwise_kl_divergence(targets, predictions)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter('../../data/experiments/graph/loss_graph', sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    print('running stuff')
    res = sess.run(result, feed_dict={targets: y, predictions: X})
    print(res)
