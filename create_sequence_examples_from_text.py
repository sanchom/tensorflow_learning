from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import pickle
import random

import tensorflow as tf

# Prepares a vocabulary and a set of training files filled with
# tf.SequenceExamples.

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('vocab', '/dev/null',
                    'Location to store the vocabulary in.')
flags.DEFINE_integer('sequence_length', 200,
                     'How long each training sequence should be.')
flags.DEFINE_integer('num_sequences', 100,
                     'How many sequence examples to extract.')
flags.DEFINE_string('output', '/dev/null',
                    'Location to store example sequences.')

def learn_vocab(paths):
  vocab = set()
  for p in paths:
    with open(p) as f:
      for line in f:
        for c in line:
          vocab.add(c)
  return sorted(list(vocab))

def get_example(data, integerization_map):
  start = random.randint(0, len(data) - FLAGS.sequence_length - 2)
  one_past_padded_end = start + FLAGS.sequence_length + 1
  padded_seq = [integerization_map[c] for c in data[start:one_past_padded_end]]
  seq = padded_seq[:FLAGS.sequence_length]
  target = padded_seq[1:]

  example = tf.train.SequenceExample()
  example.context.feature['length'].int64_list.value.append(FLAGS.sequence_length)
  
  input_tokens = example.feature_lists.feature_list['inputs']
  target_tokens = example.feature_lists.feature_list['targets']
  
  for i, t in zip(seq, target):
    input_tokens.feature.add().int64_list.value.append(i)
    target_tokens.feature.add().int64_list.value.append(t)

  return example

def save_vocab(vocab):
  with open(FLAGS.vocab, 'w') as f:
    pickle.dump(vocab, f)

def load_data(paths):
  data = ''
  for p in paths:
    with open(p) as f:
      data += f.read()
  return data

def get_reverse_map(vocab):
  return dict([(v, i) for i, v in enumerate(vocab)])

def main(argv):
  input_list = argv[1:]
  if len(input_list) < 1:
    print('No input files provided.')
    exit(1)
  vocab = learn_vocab(input_list)
  integerization_map = get_reverse_map(vocab)
  save_vocab(vocab)

  data = load_data(input_list)

  with open(FLAGS.output, 'w') as f:
    writer = tf.python_io.TFRecordWriter(f.name)
    for i in xrange(FLAGS.num_sequences):
      example = get_example(data, integerization_map)
      writer.write(example.SerializeToString())
    writer.close()
    print('Wrote {} tf.ExampleSequences to {}'.format(FLAGS.num_sequences, FLAGS.output))

if __name__ == '__main__':
  tf.app.run()
