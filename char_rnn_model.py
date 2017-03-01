from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class CharRnnModel(object):
  def __init__(self, sequence_length, cell_size, layers, vocab_size):
    self.sequence_length = sequence_length
    with tf.variable_scope('inputs'):
      self.data = tf.placeholder(tf.float32, [None, self.sequence_length, vocab_size], name='input_sequence')
      self.targets = tf.placeholder(tf.int32, [None, self.sequence_length], name='target_sequence')
    
    with tf.variable_scope('inference'):
      self.cell = tf.contrib.rnn.LSTMCell(cell_size)
      self.cell = tf.contrib.rnn.DropoutWrapper(self.cell, input_keep_prob=0.8, output_keep_prob=0.8)
      self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * layers)

      output, _ = tf.nn.dynamic_rnn(self.cell, self.data, dtype=tf.float32)

      # Flattening into a bunch of rows, each num_neurons long. Thus, each
      # output vector, at every timestep of every batch is given its own row
      # in this reshaped tensor.
      output = tf.reshape(output, [-1, cell_size])

      self.output_weights = tf.Variable(tf.truncated_normal([cell_size, vocab_size], stddev=0.1), name='output_weights')
      self.output_bias = tf.Variable(tf.constant(0.1, shape=[vocab_size]), name='output_bias')

      self.logits = tf.matmul(output, self.output_weights) + self.output_bias
      # Folding the predictions back into sequences of self.sequence_length
      self.logits = tf.reshape(self.logits, [-1, self.sequence_length, vocab_size])

    with tf.variable_scope('loss'):
      mask = tf.ones_like(self.targets, dtype=tf.float32)
      self.sequence_loss = tf.contrib.seq2seq.sequence_loss(
        self.logits, self.targets, mask)

  def loss_op(self):
    return self.sequence_loss
