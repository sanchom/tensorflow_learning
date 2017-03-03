from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import glob
import pickle
import random
import shutil
import numpy as np
import tensorflow as tf

from char_rnn_model import CharRnnModel
from sequence_pipeline import InputSequencePipeline

with open('vocabulary.pkl') as f:
  vocab = pickle.load(f)

def optimizer(loss_op, global_step):
  with tf.variable_scope('optimizer'):
    # TODO: Experiment with different learning rates.
    rate = tf.train.exponential_decay(0.01, global_step, 10000, 0.98)
    tf.summary.scalar('learning_rate', rate)
    optimize_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(
      loss_op,
      global_step=global_step)
    return optimize_op

# Define the global step and its initialization.
global_step = tf.Variable(0, name='global_step', trainable=False)

input_queue = InputSequencePipeline(len(vocab), batch_size=2)
rnn_model = CharRnnModel(
  input_queue.tensors(),
  sequence_length=200, cell_size=128, layers=2, vocab_size=len(vocab))
optimization_op = optimizer(rnn_model.loss_op(), global_step)

shutil.rmtree('/tmp/char_rnn', ignore_errors=True)
# TODO: Make the checkpoint dir a subdirectory, named based on the
# architectural hyperparameters.
with tf.train.MonitoredTrainingSession(
    checkpoint_dir="/tmp/char_rnn",
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]
) as sess:
  step = 1
  while not sess.should_stop():
    if (step % 100 == 0):
      print('Step {}'.format(step))
    sess.run(optimization_op)
    step += 1
