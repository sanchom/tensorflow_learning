from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import glob
import pickle
import os
import random
import shutil
import time
import numpy as np
import tensorflow as tf

from model import CharRnnModel
from input_pipeline import InputPipeline

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('vocabulary', '', 'Vocabulary file.')
flags.DEFINE_string('mode', 'train', '[train|sample]')
flags.DEFINE_string('checkpoint_dir', '/tmp/char_rnn',
                    '''Top-level directory under which to store specialized directories of training data.
                    Or, at sample-time, the specialized sub-directory from which to load a trained model.
                    ''')
flags.DEFINE_integer('cell_size', 64, 'How many neurons in a recursive cell.')
flags.DEFINE_integer('layers', 1, 'How many layers in the LSTM.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for training.')
flags.DEFINE_float('dropout', 0, 'Amount of dropout for training.')
flags.DEFINE_float('initial_lr', 0.01, 'Initial learning rate.')
flags.DEFINE_string('example_file', 'examples.pb', 'File with serialized tf.SequenceExamples.')

def load_vocab():
  with open(FLAGS.vocabulary) as f:
    vocab = pickle.load(f)
  return vocab

def optimizer(loss_op, global_step):
  with tf.variable_scope('optimizer'):
    rate = tf.train.exponential_decay(FLAGS.initial_lr, global_step, 1000, 0.98)
    tf.summary.scalar('learning_rate', rate)
    optimize_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(
      loss_op,
      global_step=global_step)
    return optimize_op

def train():
  specialized_checkpoint_dir = os.path.join(
    FLAGS.checkpoint_dir,
    'dropout-{}_cellsize-{}_layers-{}_lr-{}_batchsize-{}'.format(
      FLAGS.dropout, FLAGS.cell_size, FLAGS.layers, FLAGS.initial_lr, FLAGS.batch_size))
  shutil.rmtree(specialized_checkpoint_dir, ignore_errors=True)
  os.makedirs(specialized_checkpoint_dir)

  vocab = load_vocab()

  global_step = tf.Variable(0, name='global_step', trainable=False)

  # Values that need to be saved so they can be matched at sample
  # time.
  config = {}
  config['cell_size'] = FLAGS.cell_size
  config['layers'] = FLAGS.layers

  config_path = os.path.join(specialized_checkpoint_dir, 'config.pkl')
  with open(config_path, 'w') as f:
    pickle.dump(config, f)

  input_pipeline = InputPipeline([FLAGS.example_file], len(vocab), batch_size=FLAGS.batch_size)
  (input_sequence, target_sequence) = input_pipeline.tensors()
  rnn_model = CharRnnModel(
    sequence_length=200, cell_size=FLAGS.cell_size, layers=FLAGS.layers,
    vocab_size=len(vocab), dropout=FLAGS.dropout, mode='train')
  logits = rnn_model.inference(input_sequence)
  loss = rnn_model.loss(logits, target_sequence)
  optimization_op = optimizer(loss, global_step)

  class StopAtTimeHook(tf.train.SessionRunHook):
    def __init__(self, seconds):
      self._end = time.time() + seconds

    def begin(self):
      self._global_step_tensor = tf.train.get_global_step()

    def before_run(self, run_context):
      return None

    def after_run(self, run_context, run_values):
      if time.time() >= self._end:
        run_context.request_stop()

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=specialized_checkpoint_dir,
      hooks=[StopAtTimeHook(60)]
  ) as sess:
    while not sess.should_stop():
      sess.run(optimization_op)

def sample():
  vocab = load_vocab()

  config_path = os.path.join(FLAGS.checkpoint_dir, 'config.pkl')
  with open(config_path) as f:
    config = pickle.load(f)

  rnn_model = CharRnnModel(
    sequence_length=200,
    cell_size=config['cell_size'], layers=config['layers'],
    vocab_size=len(vocab), dropout=0, mode='sample')

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(tf.global_variables())
    checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
      saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
      print('Failed to load checkpoint.')
      exit(1)

    print(rnn_model.sample(sess, vocab, length=30, prime='2016'))

def main(_):
  if FLAGS.mode == 'train':
    train()
  elif FLAGS.mode == 'sample':
    sample()
  print('Exiting program...')

if __name__ == '__main__':
  tf.app.run()
