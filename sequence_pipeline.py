from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow as tf

class InputSequencePipeline(object):
  def __init__(self, sequence_length):
    with tf.variable_scope('input_queue'):
      filenames = ['whatsapp-2016-09-17.txt', 'whatsapp-2017-02-17.txt']
      filename_queue = tf.train.string_input_producer(filenames)
      reader = tf.WholeFileReader()
      key, value = reader.read(filename_queue)
      # Getting random slice of length sequence_length from this.
      example_set = tf.random_crop(value, (50))

      # Grab 10 sequence_length-length subsequences from value.

      # Create a batch of examples and targets

      # Use train.shuffle_batch with enqueue_many=True to shuffle this
      # set of examples into the training queue.
      print(example_set)

  def tensors(self):
    return (None, None)
