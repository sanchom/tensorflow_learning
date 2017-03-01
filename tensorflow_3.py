from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import glob
import random
import shutil
import numpy as np
import tensorflow as tf

from char_rnn_model import CharRnnModel

def optimizer(loss_op, global_step):
  with tf.variable_scope('optimizer'):
    rate = tf.train.exponential_decay(0.01, global_step, 10000, 0.98)
    tf.summary.scalar('learning_rate', rate)
    optimize_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(
      loss_op,
      global_step=global_step)
    return optimize_op

# Define the global step and its initialization.
global_step = tf.Variable(0, name='global_step', trainable=False)

rnn_model = CharRnnModel(sequence_length=200, cell_size=256, layers=3, vocab_size=68)
optimization_op = optimizer(rnn_model.loss_op(), global_step)
