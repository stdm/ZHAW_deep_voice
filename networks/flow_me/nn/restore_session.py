import tensorflow as tf


def restore_session(path_meta_graph, path_ckpt, output_layer_name):
    """Restores the session from given meta graph and the given checkpoint file.
    Also returns the input and output layer. Output layer can be specified in function agrs.
    """
    sess = tf.Session()
    saver = tf.train.import_meta_graph(path_meta_graph)

    saver.restore(sess, path_ckpt)

    graph = tf.get_default_graph()
    input_layer = graph.get_tensor_by_name('placeholders/input_data:0')
    output_layer = graph.get_tensor_by_name(output_layer_name)
    train_mode = graph.get_tensor_by_name('placeholders/train_mode:0')
    return sess, input_layer, output_layer, train_mode
