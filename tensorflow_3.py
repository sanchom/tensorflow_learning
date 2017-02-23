# Following https://www.tensorflow.org/community/style_guide, except I
# use 4-spaces (for now).
#
# A non-batched char-rnn model on dummy data.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import shutil
import numpy as np
import tensorflow as tf

NUM_NEURONS = 32
MAX_LENGTH = 50
VOCAB_SIZE = 27

def get_dummy_data():
    # The dummy data is just a small sequence that repeats.
    sequence = [2, 5, 9, 13, 3, 6, 1, 25, 3, 5, 9, 24]
    data = np.zeros((1, MAX_LENGTH, VOCAB_SIZE), np.float32)
    for i in xrange(MAX_LENGTH):
        data[0,i,sequence[i%len(sequence)]] = 1
    targets = np.zeros((1, MAX_LENGTH, VOCAB_SIZE), np.float32)
    # The target is just the input sequence shifted by one. I.e. the
    # network is learning to predict the next number in the sequence.
    for i in xrange(MAX_LENGTH):
        targets[0,i,sequence[(i+1)%len(sequence)]] = 1
    return data, targets

def optimizer(loss_op, global_step):
    with tf.variable_scope('optimizer'):
        rate = tf.train.exponential_decay(0.01, global_step, 1000, 0.97)
        tf.summary.scalar('learning_rate', rate)
        optimize_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(
            loss_op,
            global_step=global_step)
        return optimize_op

def inputs():
    with tf.variable_scope('inputs'):
        data = tf.placeholder(tf.float32, [1, MAX_LENGTH, VOCAB_SIZE])
        targets = tf.placeholder(tf.float32, [1, MAX_LENGTH, VOCAB_SIZE])
        return data, targets

def inference(data):
    with tf.variable_scope('rnn'):
        # Defining the recurrent cell
        cell = tf.contrib.rnn.LSTMCell(NUM_NEURONS)

        output, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

        # Flattening into a bunch of rows, each num_neurons long. Thus, each
        # output vector, at every timestep of every batch is given its own row
        # in this reshaped tensor.
        output = tf.reshape(output, [-1, NUM_NEURONS])

        output_weights = tf.Variable(tf.truncated_normal([NUM_NEURONS, VOCAB_SIZE], stddev=0.1), name='output_weights')
        output_bias = tf.Variable(tf.constant(0.1, shape=[VOCAB_SIZE]), name='output_bias')

        predictions = tf.nn.softmax(tf.matmul(output, output_weights) + output_bias)
        # Folding the predictions back into sequences of MAX_LENGTH
        predictions = tf.reshape(predictions, [-1, MAX_LENGTH, VOCAB_SIZE])
        return predictions

def loss(predictions, targets):
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(targets * tf.log(predictions), [1, 2]))
    tf.summary.scalar('loss', cross_entropy)
    return cross_entropy

# Define the global step and its initialization.
global_step = tf.Variable(0, name='global_step', trainable=False)

# Putting the graph together.
input_sequence, target_sequence = inputs()
predicted_sequence = inference(input_sequence)
loss_op = loss(predicted_sequence, target_sequence)
optimization_op = optimizer(loss_op, global_step)

# MonitoredTrainingSession automatically handles global variable
# initialization, summary writing, checkpoints, watching for stopping
# criteria, etc.
shutil.rmtree('/tmp/tensorflow_3', ignore_errors=True)
with tf.train.MonitoredTrainingSession(
        checkpoint_dir="/tmp/tensorflow_3",
        hooks=[tf.train.StopAtStepHook(last_step=100000)]
) as sess:
    step = 1
    while not sess.should_stop():
        if (step % 100 == 0):
            print('Step {}'.format(step))
        dummy_data, dummy_targets = get_dummy_data()
        sess.run(optimization_op, feed_dict={input_sequence:dummy_data, target_sequence:dummy_targets})
        step += 1
