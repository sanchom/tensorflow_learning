# Following https://www.tensorflow.org/community/style_guide, except I
# use 4-spaces (for now).
#
# Using an input queue. I made the model more complex (just more
# fully-connected layers) compared to tensorflow_1.py because the
# optimization step was just way too fast. The queue had no chance of
# keeping up with the demand for data. The TensorBoard
# fraction_of_X_full for the batching operation was always stuck near
# zero. Now, with the more expensive inference/optimization step,
# there is actually some time for the queues to fill up while the
# computation is happening.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import shutil
import numpy as np
import tensorflow as tf

BATCH_SIZE=32

def input_queue():
    with tf.variable_scope('input_queue'):
        filenames = ['dummy_data.txt']
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        col1, col2, label = tf.decode_csv(value, record_defaults=[[0.0], [0.0], [0.0]], field_delim=' ')
        example = tf.stack([col1, col2])
        min_after_dequeue = 500
        capacity = min_after_dequeue + 2 * BATCH_SIZE
        example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=BATCH_SIZE, capacity=capacity, num_threads=4,
            min_after_dequeue=min_after_dequeue)
        return example_batch, label_batch
    
def template_fc(x, input_size, output_size, name=None):
    with tf.variable_scope(name):
        w = tf.Variable(tf.random_normal([input_size,output_size]))
        tf.summary.histogram('w', w)
        b = tf.Variable(tf.constant(0.0, shape=[output_size]))
        tf.summary.histogram('b', b)
        return tf.nn.sigmoid(tf.matmul(x, w) + b)

    
def inference(x):
    # Defining the inference graph and associated summaries.
    fc1 = template_fc(x, 2, 256, 'fc1')
    fc2 = template_fc(fc1, 256, 512, 'fc2')
    fc3 = template_fc(fc2, 512, 512, 'fc3')
    fc4 = template_fc(fc3, 512, 1, 'fc4')
    return fc4

def loss(y, y_hat):
    # Defining the loss graph.
    with tf.variable_scope('loss'):
        y_hat = tf.reshape(y_hat, y.shape)
        loss_op = tf.losses.log_loss(y, y_hat)
        tf.summary.scalar('loss', loss_op)
        return loss_op

def optimizer(loss_op, global_step):
    with tf.variable_scope('optimizer'):
        rate = tf.train.exponential_decay(0.01, global_step, 1000, 0.97)
        tf.summary.scalar('learning_rate', rate)
        optimize_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(
            loss_op,
            global_step=global_step)
        return optimize_op

# Define the global step and its initialization.
global_step = tf.Variable(0, name='global_step', trainable=False)

# Putting the graph together
example_batch, label_batch = input_queue()
y_hat = inference(example_batch)
loss_op = loss(label_batch, y_hat)
optimization_op = optimizer(loss_op, global_step)

shutil.rmtree('/tmp/tensorflow_2', ignore_errors=True)

# MonitoredTrainingSession automatically handles global variable
# initialization, summary writing, checkpoints, watching for stopping
# criteria, etc.
with tf.train.MonitoredTrainingSession(
        checkpoint_dir="/tmp/tensorflow_2",
        hooks=[tf.train.StopAtStepHook(last_step=100000)]
) as sess:
    while not sess.should_stop():
        sess.run(optimization_op)
