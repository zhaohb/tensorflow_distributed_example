#
#Parameter Node (10.0.0.6)
#python mnist_tf_dist.py \
#  --job_name=ps \
#  --task_index=0
#Worker Node 1 (10.0.0.4)
#python mnist_tf_dist.py \
#  --job_name=worker \
    #  --task_index=0 --device=cpu:0/gpu:0
#Worker Node 2 (10.0.0.5)
#python mnist_tf_dist.py \
#  --job_name=worker \
    #  --task_index=1 --device=gpu:0/fpga:0
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import math

import tensorflow as tf

FLAGS = None
batch_size = 100

cluster = None
server = None
is_chief = False

def main(_):
  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:%s/task:%d/%s" % (FLAGS.job_name, FLAGS.task_index,FLAGS.device),
    cluster=cluster)):
    worker_device="/job:%s/task:%d/%s" % (FLAGS.job_name, FLAGS.task_index,FLAGS.device),
    print(worker_device)

    ###
    ### Training
    ###

    #
    # read training data
    #

    # image - 784 (=28 x 28) elements of grey-scaled integer value [0, 1]
    # label - digit (0, 1, ..., 9)
    train_queue = tf.train.string_input_producer(
      [FLAGS.train_file],
      num_epochs = 2) # data is repeated and it raises OutOfRange when data is over
    train_reader = tf.TFRecordReader()
    _, train_serialized_exam = train_reader.read(train_queue)
    train_exam = tf.parse_single_example(
      train_serialized_exam,
      features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
      })
    train_image = tf.decode_raw(train_exam['image_raw'], tf.uint8)
    train_image.set_shape([784])
    train_image = tf.cast(train_image, tf.float32) * (1. / 255)
    train_label = tf.cast(train_exam['label'], tf.int32)
    train_batch_image, train_batch_label = tf.train.batch(
      [train_image, train_label],
      batch_size=batch_size)

    #
    # define training graph
    #

    # define input
    plchd_image = tf.placeholder(
      dtype=tf.float32,
      shape=(None, 784))
    plchd_label = tf.placeholder(
      dtype=tf.int32,
      shape=(None))

    # define network and inference
    # (simple 2 fully connected hidden layer : 784->128->64->10)
    with tf.name_scope('hidden1'):
      weights = tf.Variable(
        tf.truncated_normal(
          [784, 128],
          stddev=1.0 / math.sqrt(float(784))),
        name='weights')
      biases = tf.Variable(
        tf.zeros([128]),
        name='biases')
      hidden1 = tf.nn.relu(tf.matmul(plchd_image, weights) + biases)
    with tf.name_scope('hidden2'):
      weights = tf.Variable(
        tf.truncated_normal(
          [128, 64],
          stddev=1.0 / math.sqrt(float(128))),
        name='weights')
      biases = tf.Variable(
        tf.zeros([64]),
        name='biases')
      hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    with tf.name_scope('softmax_linear'):
      weights = tf.Variable(
        tf.truncated_normal(
          [64, 10],
          stddev=1.0 / math.sqrt(float(64))),
      name='weights')
      biases = tf.Variable(
        tf.zeros([10]),
        name='biases')
      logits = tf.matmul(hidden2, weights) + biases

    # define optimization
    global_step = tf.train.create_global_step() # start without checkpoint
    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=0.07)
    loss = tf.losses.sparse_softmax_cross_entropy(
      labels=plchd_label,
      logits=logits)
    train_op = optimizer.minimize(
      loss=loss,
      global_step=global_step)

    #
    # run session
    #

    with tf.train.MonitoredTrainingSession(
      master=server.target,
      checkpoint_dir=FLAGS.out_dir,
      is_chief=is_chief) as sess:

      # when data is over, OutOfRangeError occurs and ends with MonitoredSession

      local_step_value = 0
      array_image, array_label = sess.run(
        [train_batch_image, train_batch_label])
      while not sess.should_stop():
        feed_dict = {
          plchd_image: array_image,
          plchd_label: array_label
        }
        _, global_step_value, loss_value, array_image, array_label = sess.run(
          [train_op, global_step, loss, train_batch_image, train_batch_label],
          feed_dict=feed_dict)
        local_step_value += 1
        if local_step_value % 100 == 0: # You can also use tf.train.LoggingTensorHook for output
          print("Local Step %d, Global Step %d (Loss: %.2f)" % (local_step_value, global_step_value, loss_value))

    print('training finished')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--device',
    type=str,
    default='cpu:0',
    help='Which device will be used.')
  parser.add_argument(
    '--train_file',
    type=str,
    default='/home/distributed-tensorflow-example/train.tfrecords',
    help='File path for the training data.')
  parser.add_argument(
    '--out_dir',
    type=str,
    default='/home/distributed-tensorflow-example/out',
    help='Dir path for the model and checkpoint output.')
  parser.add_argument(
    '--job_name',
    type=str,
    required=True,
    help='job name (parameter or worker) for cluster')
  parser.add_argument(
    '--task_index',
    type=int,
    required=True,
    help='index number in job for cluster')
  FLAGS, unparsed = parser.parse_known_args()

  # start server
  cluster = tf.train.ClusterSpec({
    'ps': ['127.0.0.1:2222'],
    'worker': [
      '127.0.0.1:3333',
      '127.0.0.1:4444'
    ]})
  server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    is_chief = (FLAGS.task_index == 0)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
