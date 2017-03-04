from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import tensorflow as tf

class InputPipeline(object):
  def __init__(self, example_paths, vocab_depth, batch_size):
    with tf.variable_scope('input_queue'):
      filename_queue = tf.train.string_input_producer(example_paths)
      reader = tf.TFRecordReader()
      key, value = reader.read(filename_queue)

      # Definining how to parse the example. This has to match the
      # format in create_sequence_examples_from_text.py.
      context_features = {
        'length': tf.FixedLenFeature([], dtype=tf.int64)
      }
      sequence_features = {
        'inputs': tf.FixedLenSequenceFeature([], dtype=tf.int64),
        'targets': tf.FixedLenSequenceFeature([], dtype=tf.int64)
      }

      context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=value,
        context_features=context_features,
        sequence_features=sequence_features
      )

      example = tf.one_hot(sequence_parsed['inputs'], vocab_depth, name='input_sequence')
      targets = tf.identity(sequence_parsed['targets'], name='targets')

      # TODO: Remove this hard-coded length (200), and handle
      # variable-length sequences.
      # TODO: Figure out how to use batch_sequences_with_states.
      min_after_dequeue = 128
      capacity = min_after_dequeue + 2 * batch_size
      self.example_batch, self.targets_batch = tf.train.shuffle_batch(
        [example, targets], shapes=([200, vocab_depth], [200]),
        batch_size=batch_size, capacity=capacity, num_threads=4,
        min_after_dequeue=min_after_dequeue)

  def tensors(self):
    return self.example_batch, self.targets_batch
