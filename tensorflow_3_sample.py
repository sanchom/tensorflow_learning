# Following https://www.tensorflow.org/community/style_guide, except I
# use 4-spaces (for now).
#
# A non-batched char-rnn model on dummy data.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import glob
import random
import shutil
import numpy as np
import tensorflow as tf

NUM_NEURONS = 128
MAX_LENGTH = 200
BATCH_SIZE = 16

class Vocab(object):
    def __init__(self):
        self.data = ''
        for p in glob.glob('/Users/sanchom/Desktop/whatsapp*'):
            with open(p) as f:
                self.data += f.read()
        self.vocab = sorted(list(set(self.data)))
        self.vocab_size = len(self.vocab)

def inference_pieces(vocab):
    with tf.variable_scope('rnn'):
        # Defining the recurrent cell
        cell = tf.contrib.rnn.LSTMCell(NUM_NEURONS)

        output_weights = tf.Variable(tf.truncated_normal([NUM_NEURONS, vocab.vocab_size], stddev=0.1), name='output_weights')
        output_bias = tf.Variable(tf.constant(0.1, shape=[vocab.vocab_size]), name='output_bias')
        
        return cell, output_weights, output_bias

vocab = Vocab()

# Getting the trained bits
cell, weights, bias = inference_pieces(vocab)

# Defining the input placeholders
input_char = tf.placeholder(tf.float32, [1, vocab.vocab_size])
input_state = cell.zero_state(1, tf.float32)

cell_output, next_state = cell(input_char, input_state)
cell_output = tf.nn.softmax(tf.matmul(cell_output, weights) + bias)
output_and_state = [cell_output, next_state]

# This sampling routine is taken from
# https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/model.py
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # Somehow, I didn't recreate the graph the same way as in
    # tensorflow_3.py. This section just changes the variable keys so
    # that it can load from the checkpoint file.
    names_to_vars = {v.op.name: v for v in tf.global_variables()}
    weights_var = names_to_vars['lstm_cell/weights']
    names_to_vars['rnn/rnn/lstm_cell/weights'] = weights_var
    del names_to_vars['lstm_cell/weights']
    bias_var = names_to_vars['lstm_cell/biases']
    names_to_vars['rnn/rnn/lstm_cell/biases'] = bias_var
    del names_to_vars['lstm_cell/biases']
    
    saver = tf.train.Saver(var_list=names_to_vars)
    ckpt = tf.train.get_checkpoint_state('/tmp/tensorflow_3')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

        state = sess.run(cell.zero_state(1, tf.float32))
        prime = '2016'

        for char in prime[:-1]:
            x = np.zeros((1, vocab.vocab_size))
            x[0, vocab.vocab.index(char)] = 1
            feed = {input_char: x, input_state: state}
            [output, state] = sess.run(output_and_state,
                                       feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(10000):
            x = np.zeros((1, vocab.vocab_size))
            x[0, vocab.vocab.index(char)] = 1
            feed = {input_char: x, input_state:state}
            [probs, state] = sess.run(output_and_state, feed)
            p = probs[0]
            sample = weighted_pick(p)
            pred = vocab.vocab[sample]
            ret += pred
            char = pred

        print(ret)
