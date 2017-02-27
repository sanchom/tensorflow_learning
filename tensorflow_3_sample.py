# Following https://www.tensorflow.org/community/style_guide, except I
# use 4-spaces (for now).
#
# The sampler for the char-rnn trained with tensorflow_3.py.
#
# TODO: Make sampler and trainer use a common model, to remove
# redundancy, and make loading checkpoint files work without renaming
# keys.
#
# TODO: Have trainer save vocab and architecture parameters to remove
# hard-coded dependencies.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import glob
import random
import shutil
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('length', 30, 'Number of lines to generate.')
flags.DEFINE_string('checkpoint_dir', '/tmp',
                    'The checkpoint is in this directory.')
flags.DEFINE_string('prime', '2016',
                    'Text that will prime the sequence.')

NUM_NEURONS = 256
NUM_LAYERS = 3

class Vocab(object):
    def __init__(self):
        self.vocab = ['\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|', '~', '\x80', '\x81', '\x82', '\x83', '\x84', '\x85', '\x86', '\x87', '\x88', '\x89', '\x8a', '\x8b', '\x8c', '\x8d', '\x8e', '\x8f', '\x90', '\x91', '\x92', '\x93', '\x94', '\x95', '\x96', '\x97', '\x98', '\x99', '\x9a', '\x9b', '\x9c', '\x9d', '\x9e', '\x9f', '\xa0', '\xa2', '\xa3', '\xa4', '\xa5', '\xa6', '\xa7', '\xa8', '\xa9', '\xaa', '\xab', '\xac', '\xad', '\xae', '\xaf', '\xb0', '\xb1', '\xb2', '\xb3', '\xb4', '\xb5', '\xb6', '\xb7', '\xb8', '\xb9', '\xba', '\xbb', '\xbc', '\xbd', '\xbe', '\xbf', '\xc2', '\xc3', '\xc5', '\xe2', '\xe3', '\xef', '\xf0']
        self.vocab_size = len(self.vocab)

def inference_pieces(vocab):
    with tf.variable_scope('rnn'):
        # Defining the recurrent cell
        cell = tf.contrib.rnn.LSTMCell(NUM_NEURONS)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * NUM_LAYERS)

        output_weights = tf.Variable(tf.truncated_normal([NUM_NEURONS, vocab.vocab_size], stddev=0.1), name='output_weights')
        output_bias = tf.Variable(tf.constant(0.1, shape=[vocab.vocab_size]), name='output_bias')
        
        return cell, output_weights, output_bias

def main(argv=None):
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
        for c in xrange(NUM_LAYERS):
            for vname in ['weights', 'biases']:
                saved_name = 'rnn/rnn/multi_rnn_cell/cell_{}/lstm_cell/{}'.format(c, vname)
                our_name = 'multi_rnn_cell/cell_{}/lstm_cell/{}'.format(c, vname)
                var = names_to_vars[our_name]
                names_to_vars[saved_name] = var
                del names_to_vars[our_name]

        saver = tf.train.Saver(var_list=names_to_vars)
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

            state = sess.run(cell.zero_state(1, tf.float32))
            prime = FLAGS.prime

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
            while (len(ret.split('\n')) < FLAGS.length):
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

if __name__ == '__main__':
      tf.app.run()
