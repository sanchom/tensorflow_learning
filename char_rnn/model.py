from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class CharRnnModel(object):
  def __init__(self, sequence_length, cell_size, layers, vocab_size, dropout, mode):
    assert(mode == 'train' or mode == 'sample')
    assert(dropout == 0 or mode == 'train')
    self.sequence_length = sequence_length
    self.cell_size = cell_size
    self.vocab_size = vocab_size

    with tf.variable_scope('inference'):
      # Note: These commands don't create tensors or ops. Simply
      # defining the form of the cell doesn't make tensorflow create
      # trainable variables for these. That happens by calling
      # tf.nn.dynamic_rnn(...) or by otherwise using this cell as part
      # of the graph.
      self.cell = tf.contrib.rnn.LSTMCell(self.cell_size)
      self.cell = tf.contrib.rnn.DropoutWrapper(
        self.cell, input_keep_prob=(1-dropout), output_keep_prob=(1-dropout))
      self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * layers)

      self.output_weights = tf.Variable(tf.truncated_normal([self.cell_size, self.vocab_size], stddev=0.1), name='output_weights')
      self.output_bias = tf.Variable(tf.constant(0.1, shape=[self.vocab_size]), name='output_bias')

    if mode == 'sample':
      self._define_sample_elements()

  def inference(self, input_sequence):
    with tf.variable_scope('inference'):
      # TODO: Use a fused RNN for faster GPU computation.

      # Note: dynamic_rnn creates a name scope 'rnn', within which the
      # internal LSTM weights and biases are located. Thus, the name
      # of the LSTM weights and biases are prefixed by
      # 'inference/rnn'. When loading at sample time, you have to be
      # sure to match this prefix manually, since you likely won't be
      # calling dynamic_rnn.
      output, _ = tf.nn.dynamic_rnn(self.cell, input_sequence, dtype=tf.float32)
      # Flattening into a bunch of rows, each num_neurons long. Thus, each
      # output vector, at every timestep of every batch is given its own row
      # in this reshaped tensor.
      output = tf.reshape(output, [-1, self.cell_size])
      logits = tf.matmul(output, self.output_weights) + self.output_bias
      # Unfolding the predictions back into sequences of self.sequence_length.
      logits = tf.reshape(logits, [-1, self.sequence_length, self.vocab_size])
      return logits

  def loss(self, logits, target_sequence):
    with tf.variable_scope('loss'):
      mask = tf.ones_like(target_sequence, dtype=tf.float32)
      sequence_loss = tf.contrib.seq2seq.sequence_loss(logits, target_sequence, mask)
      tf.summary.scalar('loss', sequence_loss)
      return sequence_loss

  def _define_sample_elements(self):
    # Explicitly naming this scope 'inference/rnn' to match the
    # implicitly created 'rnn' scope when using dynamic_rnn during
    # training inference.
    with tf.variable_scope('inference/rnn'):
      # TODO: Switch to new dynamic_decoder.
      self.input_char = tf.placeholder(tf.float32, [1, self.vocab_size])
      self.input_state = self.cell.zero_state(1, tf.float32)

      self.cell_output, self.next_state = self.cell(self.input_char, self.input_state)
      self.cell_output = tf.nn.softmax(tf.matmul(self.cell_output, self.output_weights) +
                                       self.output_bias)
      self.output_and_state = [self.cell_output, self.next_state]

  # TODO: Consider removing actual vocab from this call, and performing
  # integerization on the calling side.
  def sample(self, sess, vocab, length=30, prime='2016'):
    assert(len(vocab) == self.vocab_size)
    integerization_map = dict([(v, i) for i, v in enumerate(vocab)])

    state = sess.run(self.cell.zero_state(1, tf.float32))

    for char in prime[:-1]:
      x = np.zeros((1, self.vocab_size))
      x[0, integerization_map[char]] = 1
      feed = {self.input_char: x, self.input_state: state}
      [self.output, self.state] = sess.run(self.output_and_state, feed)

    def weighted_pick(weights):
      t = np.cumsum(weights)
      s = np.sum(weights)
      return(int(np.searchsorted(t, np.random.rand(1)*s)))

    sampled_sequence = prime
    char = prime[-1]
    while (len(sampled_sequence.split('\n')) < length):
      x = np.zeros((1, self.vocab_size))
      x[0, integerization_map[char]] = 1
      feed = {self.input_char: x, self.input_state:state}
      [probs, self.state] = sess.run(self.output_and_state, feed)
      p = probs[0]
      prediction = vocab[weighted_pick(p)]
      sampled_sequence += prediction
      char = prediction

    return sampled_sequence
