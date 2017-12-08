"""
    A file that copies the essential part of the network training in a function.

    Based on work of Gygax and Egly.
"""
import shutil
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from common.utils.load_config import *
from common.utils.logger import *
from common.utils.paths import *
from . import DataGen


def train_network(config, clear, debug, nchw):
    # Read config
    path_master_config = get_configs('master')
    config = load_config(path_master_config, config)

    flow_me = "flow_me"
    flow_me_logs = get_experiment_logs(flow_me)
    flow_me_nets = get_experiment_nets(flow_me)

    if clear:
        try:
            shutil.rmtree(flow_me_logs)
        except OSError:
            pass

    if debug:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        logger = get_logger('network', logging.DEBUG)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Filter warnings out
        logger = get_logger('network', logging.INFO)

    add_file_handler(logger, get_experiment_logs(flow_me, "training.log"))

    if not check_config(config):
        logger.critical('Config file {0} seems not valid.'.format(config.get('exp', 'name')))
        logger.info('Check if logfiles already exists or cannot be deleted.')
        sys.exit('Exit')

    logger.info('Experiment: \t' + config.get('exp', 'name'))
    logger.info('Description:\t' + config.get('exp', 'desc'))

    # Set parameters for network for nhwc oder nchw
    if nchw:
        data_format = 'channels_first'
        norm_axis = 1
        shape_ph_input_data = (
            None, 1, config.getint('spectrogram', 'frequency_elements'), config.getint('spectrogram', 'duration'))
    else:
        data_format = 'channels_last'
        norm_axis = 3
        shape_ph_input_data = (
            None, config.getint('spectrogram', 'frequency_elements'), config.getint('spectrogram', 'duration'), 1)

    logger.info("Starting to build Network")

    # placeholder
    with tf.name_scope('placeholders'):
        ph_input_data = tf.placeholder(tf.float32, shape=shape_ph_input_data, name='input_data')
        ph_map_labels = tf.placeholder(tf.int32, shape=(config.getint('net', 'batch_size')), name='map_labels')
        ph_class_input_data = tf.placeholder(tf.float32, shape=(config.getint(
            'train', 'total_speakers'), config.getfloat('net', 'dense10_factor') * config.getint('train',
                                                                                                 'total_speakers')),
                                             name='class_input_data')
        ph_train_mode = tf.placeholder(tf.bool, shape=[], name='train_mode')

        if debug:
            tf.summary.histogram('input', ph_input_data)
            if not nchw:
                tf.summary.image('input_image', ph_input_data, max_outputs=3)

    # L1: convolution (#32)
    with tf.name_scope('l1_conv'):
        l1 = tf.layers.conv2d(ph_input_data, filters=config.getint('net', 'conv1_filter'), kernel_size=[config.getint(
            'net', 'conv_kernel'), config.getint('net', 'conv_kernel')], padding=config.get('net', 'conv_pad'),
                              data_format=data_format, activation=tf.nn.relu)

    # L2: batch-norm
    with tf.name_scope('l2_batch'):
        if config.getboolean('net', 'norm_on'):
            l2 = tf.layers.batch_normalization(l1, axis=norm_axis, epsilon=config.getfloat('net', 'norm_eps'),
                                               momentum=config.getfloat('net', 'norm_mom'), training=ph_train_mode)
        else:
            l2 = l1

    # L3: max-pooling (4x4)
    with tf.name_scope('l3_max_pooling'):
        l3 = tf.layers.max_pooling2d(inputs=l2, pool_size=[config.getint('net', 'pool_size'), config.getint(
            'net', 'pool_size')], strides=config.getint('net', 'pool_strides'), data_format=data_format)

    # L4: convolution (#64)
    with tf.name_scope('l4_conv'):
        l4 = tf.layers.conv2d(inputs=l3, filters=config.getint('net', 'conv4_filter'), kernel_size=[config.getint(
            'net', 'conv_kernel'), config.getint('net', 'conv_kernel')], padding=config.get('net', 'conv_pad'),
                              data_format=data_format, activation=tf.nn.relu)

    # L5: batch-norm
    with tf.name_scope('l5_batch'):
        if config.getboolean('net', 'norm_on'):
            l5 = tf.layers.batch_normalization(l4, axis=norm_axis, epsilon=config.getfloat('net', 'norm_eps'),
                                               momentum=config.getfloat('net', 'norm_mom'), training=ph_train_mode)
        else:
            l5 = l4

    # L6: max-pooling (4x4)
    with tf.name_scope('l6_max_pooling'):
        l6 = tf.layers.max_pooling2d(inputs=l5, pool_size=[config.getint('net', 'pool_size'), config.getint(
            'net', 'pool_size')], strides=config.getint('net', 'pool_strides'), data_format=data_format)

    # L6: reshape after last conv
    with tf.name_scope('l6_reshape'):
        dim_flatted = int(np.prod(l6.shape[1:]))
        l6 = tf.reshape(l6, [-1, dim_flatted])

    # L7: dense
    with tf.name_scope('l7_dense'):
        l7 = tf.layers.dense(inputs=l6, units=(config.getfloat('net', 'dense7_factor') *
                                               config.getint('train', 'total_speakers')), activation=tf.nn.relu)

    # L8: batch-norm
    with tf.name_scope('l8_batch'):
        if config.getboolean('net', 'norm_on'):
            l8 = tf.layers.batch_normalization(l7, axis=1, epsilon=config.getfloat('net', 'norm_eps'),
                                               momentum=config.getfloat('net', 'norm_mom'), training=ph_train_mode)
        else:
            l8 = l7
    # L9: dropout
    with tf.name_scope('l9_dropout'):
        l9 = tf.layers.dropout(inputs=l8, rate=config.getfloat('net', 'dropout_rate'), training=ph_train_mode)

    # L10: dense
    with tf.name_scope('l10_dense'):
        l10 = tf.layers.dense(inputs=l9, units=(config.getfloat('net', 'dense10_factor') *
                                                config.getint('train', 'total_speakers')), activation=tf.nn.relu)

    # L11: dense
    with tf.name_scope('l11_dense'):
        l11 = tf.layers.dense(inputs=l10, units=(config.getfloat('net', 'dense11_factor') *
                                                 config.getint('train', 'total_speakers')), activation=None)

    # loss
    with tf.name_scope('loss'):
        loss = tf.map_fn(lambda x: tf.add(tf.norm(tf.subtract(x[0], ph_class_input_data[x[1]]), ord='euclidean'),
                                          tf.log(tf.add(1e-6, tf.reduce_sum(
                                              tf.map_fn(lambda z: tf.exp(tf.negative(tf.norm(tf.subtract(x[0], z),
                                                                                             ord='euclidean'))),
                                                        ph_class_input_data), 0)))), (l10, ph_map_labels),
                         dtype=tf.float32)

        loss = tf.reduce_mean(loss)

        tf.summary.scalar('loss', loss)

    if debug:
        with tf.name_scope('debug'):
            tf.summary.histogram('emb', l10)
            all_vars = tf.trainable_variables()
            for var in all_vars:
                tf.summary.histogram(var.name, var)

    # optimizer
    with tf.name_scope('optimizer'):
        if config.get('optimizer', 'name') == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=config.getfloat('optimizer', 'learning_rate'),
                                               beta1=config.getfloat(
                                                   'optimizer', 'beta1'), beta2=config.getfloat('optimizer', 'beta2'),
                                               epsilon=config.getfloat('optimizer', 'eps'))
        elif config.get('optimizer', 'name') == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=config.getfloat(
                'optimizer', 'learning_rate'), rho=config.getfloat('optimizer', 'rho'),
                epsilon=config.getfloat('optimizer', 'eps'))
        else:
            optimizer = tf.train.MomentumOptimizer(learning_rate=config.getfloat(
                'optimizer', 'learning_rate'), momentum=config.getfloat('optimizer', 'momentum'), use_nesterov=True)
        train = optimizer.minimize(loss)

    logger.info("Finished building network")

    # Init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # TensorBoard ops
    summary = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=1000)
    train_writer = tf.summary.FileWriter(flow_me_logs, sess.graph)

    # load training and test data
    train_data_gen = DataGen.DataGen(config, data_set='train')
    validation_data_gen = DataGen.DataGen(config, data_set='validation')

    # training loop
    for step in range(config.getint('net', 'max_iter')):
        start_time = time.time()

        input_data_train, class_input_data_train, map_labels = train_data_gen.create_batch()

        if not nchw:
            input_data_train = np.transpose(input_data_train, axes=(0, 2, 3, 1))
            class_input_data_train = np.transpose(class_input_data_train, axes=(0, 2, 3, 1))

        # run z
        # it has to be ph_input_data, seems strange, but is right
        class_embeddings = sess.run(l10, feed_dict={ph_input_data: class_input_data_train, ph_train_mode: False})

        # run x
        _, loss_value = sess.run([train, loss], feed_dict={ph_input_data: input_data_train,
                                                           ph_map_labels: map_labels,
                                                           ph_class_input_data: class_embeddings,
                                                           ph_train_mode: True})

        duration = time.time() - start_time
        logger.info('Step {}: loss = {:.4f} ({:.3f} sec)'.format(step, loss_value, duration))

        # Write the summaries and print an overview every x-th step.
        if step % config.getint('net', 'sum_iter') == 0:
            # Print status

            start_time = time.time()
            # Update events file
            summary_str = sess.run(summary, feed_dict={ph_input_data: input_data_train,
                                                       ph_map_labels: map_labels,
                                                       ph_class_input_data: class_embeddings,
                                                       ph_train_mode: False})

            train_writer.add_summary(summary_str, step)
            train_writer.flush()

            duration = time.time() - start_time
            logger.debug('Step {}: Summary ({:.3f} sec)'.format(step, duration))

        # Save a checkpoint and evaluate the model periodically.
        if (step + 1) % config.getint('net', 'chkp_iter') == 0 or (step + 1) == config.getint('net', 'max_iter'):
            start_time = time.time()

            step_folder = 'step_{:05d}'.format(step)
            step_path = join(flow_me_logs, step_folder)
            test_writer = tf.summary.FileWriter(step_path, sess.graph)

            # Evaluate against the training set.
            input_data_test, map_labels = validation_data_gen.get_random_samples(config.getint('validation', 'samples'))

            if not nchw:
                input_data_test = np.transpose(input_data_test, axes=(0, 2, 3, 1))
            labels = validation_data_gen.get_labels()

            embeddings_test = sess.run(l10, feed_dict={ph_input_data: input_data_test, ph_train_mode: False})

            # Embeddings
            emb_var = tf.Variable(embeddings_test, name='embeddings')
            init_vars = tf.variables_initializer([emb_var])
            sess.run(init_vars)

            projector_config = projector.ProjectorConfig()
            embedding = projector_config.embeddings.add()
            embedding.tensor_name = emb_var.name

            # Save metafile with labels and link it to tensorboard.
            metafile_path = join(step_path, 'metadata.tsv')
            metafile_dir = os.path.dirname(metafile_path)
            if not os.path.exists(metafile_dir):
                os.makedirs(metafile_dir)

            with open(metafile_path, mode='w+') as f:
                for label_no in map_labels:
                    f.write(str(labels[label_no]) + '\n')

            embedding.metadata_path = metafile_path
            projector.visualize_embeddings(test_writer, projector_config)
            emb_saver = tf.train.Saver([emb_var])

            emb_saver.save(sess, join(step_path, 'emb.ckpt'), global_step=step)

            # Checkpoint
            saver.save(sess, join(flow_me_nets, 'model.ckpt'), global_step=step)

            duration = time.time() - start_time
            logger.info('Step {}: Checkpoint and Evaluation ({:.3f} sec)'.format(step, duration))

    sess.close()
