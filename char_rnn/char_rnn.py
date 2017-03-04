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

from model import CharRnnModel
from input_pipeline import InputPipeline

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('vocabulary', '', 'Vocabulary file.')
flags.DEFINE_string('mode', 'train', '[train|sample]')
flags.DEFINE_string('checkpoint_dir', '/tmp/char_rnn',
                    'Where to store training data.')

def load_vocab():
  with open(FLAGS.vocabulary) as f:
    vocab = pickle.load(f)
  return vocab

def optimizer(loss_op, global_step):
  with tf.variable_scope('optimizer'):
    # TODO: Experiment with different learning rates.
    rate = tf.train.exponential_decay(0.01, global_step, 10000, 0.98)
    tf.summary.scalar('learning_rate', rate)
    optimize_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(
      loss_op,
      global_step=global_step)
    return optimize_op

def train():
  vocab = load_vocab()

  global_step = tf.Variable(0, name='global_step', trainable=False)

  input_pipeline = InputSequencePipeline(len(vocab), batch_size=2)
  (input_sequence, target_sequence) = input_pipeline.tensors()
  rnn_model = CharRnnModel(
    sequence_length=200, cell_size=128, layers=2, vocab_size=len(vocab), dropout=0.2, mode='train')
  logits = rnn_model.inference(input_sequence)
  loss = rnn_model.loss(logits, target_sequence)
  optimization_op = optimizer(loss, global_step)

  shutil.rmtree(FLAGS.checkpoint_dir, ignore_errors=True)
  # TODO: Make the checkpoint dir a subdirectory, named based on the
  # architectural hyperparameters.
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.checkpoint_dir,
      hooks=[tf.train.StopAtStepHook(last_step=1000000)]
  ) as sess:
    step = 1
    while not sess.should_stop():
      if (step % 100 == 0):
        print('Step {}'.format(step))
      sess.run(optimization_op)
      step += 1

def sample():
  vocab = load_vocab()

  # TODO: Load this from a configuration that's saved during training.
  rnn_model = CharRnnModel(
    sequence_length=200, cell_size=128, layers=2, vocab_size=len(vocab), dropout=0, mode='sample')

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

if __name__ == '__main__':
  tf.app.run()
