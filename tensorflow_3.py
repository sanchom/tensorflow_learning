from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import glob
import random
import shutil
import numpy as np
import tensorflow as tf

NUM_NEURONS = 256
MAX_LENGTH = 200
BATCH_SIZE = 8
NUM_LAYERS = 3

class Data(object):
  def __init__(self):
    self.data = ''
    for p in glob.glob('/Users/sanchom/Desktop/whatsapp*'):
      with open(p) as f:
        self.data += f.read()
    self.vocab = sorted(list(set(self.data)))
    self.vocab_size = len(self.vocab)
    self.int_data = []
    for d in self.data:
      self.int_data.append(self.vocab.index(d))

  def get_next_batch(self):
    data = np.zeros((BATCH_SIZE, MAX_LENGTH, self.vocab_size), np.float32)
    starting_ids = []
    for b in xrange(BATCH_SIZE):
      starting_ids.append(random.randint(0, len(self.data) - MAX_LENGTH - 2))
    for b in xrange(BATCH_SIZE):
      for i in xrange(MAX_LENGTH):
        data[b,i,self.int_data[starting_ids[b] + i]] = 1
    targets = np.zeros((BATCH_SIZE, MAX_LENGTH), np.int32)
    # The target is just the input sequence shifted by one. I.e. the
    # network is learning to predict the next number in the sequence.
    # But, the targets are represented simply by integers, not 1-of-k.
    for b in xrange(BATCH_SIZE):
      for i in xrange(MAX_LENGTH):
        targets[b,i] = self.int_data[starting_ids[b] + i + 1]
    return data, targets

def optimizer(loss_op, global_step):
  with tf.variable_scope('optimizer'):
    rate = tf.train.exponential_decay(0.01, global_step, 10000, 0.98)
    tf.summary.scalar('learning_rate', rate)
    optimize_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(
      loss_op,
      global_step=global_step)
    return optimize_op

def inputs(vocab_size):
  with tf.variable_scope('inputs'):
    data = tf.placeholder(tf.float32, [None, MAX_LENGTH, vocab_size])
    targets = tf.placeholder(tf.int32, [None, MAX_LENGTH])
    return data, targets

# Vocab size can be inferred from data
def inference(data, vocab_size):
  with tf.variable_scope('rnn'):
    # Defining the recurrent cell
    cell = tf.contrib.rnn.LSTMCell(NUM_NEURONS)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=0.9, output_keep_prob=0.9)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS)

    output, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

    # Flattening into a bunch of rows, each num_neurons long. Thus, each
    # output vector, at every timestep of every batch is given its own row
    # in this reshaped tensor.
    output = tf.reshape(output, [-1, NUM_NEURONS])

    output_weights = tf.Variable(tf.truncated_normal([NUM_NEURONS, vocab_size], stddev=0.1), name='output_weights')
    output_bias = tf.Variable(tf.constant(0.1, shape=[vocab_size]), name='output_bias')

    logits = tf.matmul(output, output_weights) + output_bias
    # Folding the predictions back into sequences of MAX_LENGTH
    logits = tf.reshape(logits, [-1, MAX_LENGTH, vocab_size])
    return logits

def loss(logits, targets):
  weights = tf.ones_like(targets, dtype=tf.float32)
  sequence_loss = tf.contrib.seq2seq.sequence_loss(logits, targets, weights)
  tf.summary.scalar('loss', sequence_loss)
  return sequence_loss

data = Data()

# Define the global step and its initialization.
global_step = tf.Variable(0, name='global_step', trainable=False)

# Putting the graph together.
input_sequence, target_sequence = inputs(data.vocab_size)
logit_sequence = inference(input_sequence, data.vocab_size)
loss_op = loss(logit_sequence, target_sequence)
optimization_op = optimizer(loss_op, global_step)

# MonitoredTrainingSession automatically handles global variable
# initialization, summary writing, checkpoints, watching for stopping
# criteria, etc.
shutil.rmtree('/tmp/tensorflow_3', ignore_errors=True)
with tf.train.MonitoredTrainingSession(
    checkpoint_dir="/tmp/tensorflow_3",
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]
) as sess:
  step = 1
  while not sess.should_stop():
    if (step % 100 == 0):
      print('Step {}'.format(step))
    dummy_data, dummy_targets = data.get_next_batch()
    sess.run(optimization_op, feed_dict={input_sequence:dummy_data, target_sequence:dummy_targets})
    step += 1
