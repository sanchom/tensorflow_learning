from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import shutil
import numpy as np
import tensorflow as tf

class Data(object):
    def __init__(self):
        self.index = 0
        self.data = []
        with open('dummy_data.txt') as f:
            for line in f:
                self.data.append(tuple([float(t) for t in line.split()]))

    def get_next_batch(self, batch_size):
        x = np.zeros((batch_size,2), np.float32)
        y = np.zeros((batch_size,1), np.float32)
        for i in xrange(batch_size):
            x[i,0] = self.data[self.index][0]
            x[i,1] = self.data[self.index][1]
            y[i,0] = self.data[self.index][2]
            self.index = (self.index + 1) % len(self.data)
        return x, y

def inputs():
    # Defining the input placeholders.
    with tf.variable_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=(None, 2)) # 2-d vector
        y = tf.placeholder(tf.float32, shape=(None, 1)) # scalar
        return x, y

def inference(x):
    # Defining the inference graph and associated summaries.
    with tf.variable_scope('fully_connected'):
        w = tf.Variable(tf.random_normal([2,1]))
        tf.summary.histogram('w', w)
        b = tf.Variable(tf.constant(0.0))
        tf.summary.histogram('b', b)
        return tf.nn.sigmoid(tf.matmul(x, w) + b)

def loss(y, y_hat):
    # Defining the loss graph.
    with tf.variable_scope('loss'):
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
x, y = inputs()
y_hat = inference(x)
loss_op = loss(y, y_hat)
optimization_op = optimizer(loss_op, global_step)

data = Data()

shutil.rmtree('/tmp/tensorflow_1', ignore_errors=True)

# MonitoredTrainingSession automatically handles global variable
# initialization, summary writing, checkpoints, watching for stopping
# criteria, etc.
with tf.train.MonitoredTrainingSession(
        checkpoint_dir="/tmp/tensorflow_1",
        hooks=[tf.train.StopAtStepHook(last_step=100000)]
) as sess:
    while not sess.should_stop():
        input_x, input_y = data.get_next_batch(100)
        sess.run(optimization_op, feed_dict={x: input_x, y: input_y})
