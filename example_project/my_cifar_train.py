# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Training the model"""

__author__ = 'HANEL'

import os.path
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from six.moves import xrange

import imageflow
# from imageflow import inputs
# from imageflow import distorted_inputs
import my_cifar



# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 50000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', 'tmp/my-model',
                           """Directory where to write model proto """
                           """ to import in c++""")
tf.app.flags.DEFINE_string('train_dirr', 'tmp/log',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('eval_dir', 'tmp/log_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'tmp/ckpt',
                           """Directory where to read model checkpoints.""")

# Parameters
display_step = 1
val_step = 5
save_step = 50
IMAGE_PIXELS = 32 * 32 * 3
NEW_LINE = '\n'


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded ckpt in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test ckpt sets.
  # batch_size = -1
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         IMAGE_PIXELS))
  # 32, 32, 3))
  labels_placeholder = tf.placeholder(tf.int32, shape=batch_size)

  return images_placeholder, labels_placeholder


def train(re_train=True):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

    # Get images and labels for CIFAR-10.
    # images, labels = my_input.inputs()
    images, labels = imageflow.distorted_inputs(filename='../my_data_raw/train.tfrecords', batch_size=FLAGS.batch_size,
                                      num_epochs=FLAGS.num_epochs,
                                      num_threads=5, imshape=[32, 32, 3], imsize=32)
    val_images, val_labels = imageflow.inputs(filename='../my_data_raw/validation.tfrecords', batch_size=FLAGS.batch_size,
                                    num_epochs=FLAGS.num_epochs,
                                    num_threads=5, imshape=[32, 32, 3])

    print (images.get_shape(), val_images.get_shape())
    # Build a Graph that computes the logits predictions from the inference model.
    logits = my_cifar.inference(images_placeholder)

    # Calculate loss.
    loss = my_cifar.loss(logits, labels_placeholder)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = my_cifar.training(loss, global_step)

    # Calculate accuracy #
    acc, n_correct = my_cifar.evaluation(logits, labels_placeholder)

    # Create a saver.
    saver = tf.train.Saver()

    tf.scalar_summary('Acc', acc)
    # tf.scalar_summary('Val Acc', acc_val)
    tf.scalar_summary('Loss', loss)
    tf.image_summary('Images', tf.reshape(images, shape=[-1, 32, 32, 3]), max_images=10)
    tf.image_summary('Val Images', tf.reshape(val_images, shape=[-1, 32, 32, 3]), max_images=10)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    # NUM_CORES = 2  # Choose how many cores to use.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, ))
    # inter_op_parallelism_threads=NUM_CORES,
    # intra_op_parallelism_threads=NUM_CORES))
    sess.run(init)

    # Write all terminal output results here
    val_f = open("tmp/val.txt", "ab")

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dirr,
                                            graph=sess.graph)

    if re_train:

      # Export graph to import it later in c++
      # tf.train.write_graph(sess.graph, FLAGS.model_dir, 'train.pbtxt') # TODO: uncomment to get graph and use in c++

      continue_from_pre = False

      if continue_from_pre:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
        print ckpt.model_checkpoint_path
        if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)
          print('Session Restored!')

      try:
        while not coord.should_stop():

          for step in xrange(FLAGS.max_steps):

            images_r, labels_r = sess.run([images, labels])
            images_val_r, labels_val_r = sess.run([val_images, val_labels])

            train_feed = {images_placeholder: images_r,
                          labels_placeholder: labels_r}

            val_feed = {images_placeholder: images_val_r,
                        labels_placeholder: labels_val_r}

            start_time = time.time()

            _, loss_value = sess.run([train_op, loss], feed_dict=train_feed)
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % display_step == 0:
              num_examples_per_step = FLAGS.batch_size
              examples_per_sec = num_examples_per_step / duration
              sec_per_batch = float(duration)

              format_str = ('%s: step %d, loss = %.6f (%.1f examples/sec; %.3f '
                            'sec/batch)')
              print_str_loss = format_str % (datetime.now(), step, loss_value,
                                             examples_per_sec, sec_per_batch)
              print (print_str_loss)
              val_f.write(print_str_loss + NEW_LINE)
              summary_str = sess.run([summary_op], feed_dict=train_feed)
              summary_writer.add_summary(summary_str[0], step)

            if step % val_step == 0:
              acc_value, num_corroect = sess.run([acc, n_correct], feed_dict=train_feed)

              format_str = '%s: step %d,  train acc = %.2f, n_correct= %d'
              print_str_train = format_str % (datetime.now(), step, acc_value, num_corroect)
              val_f.write(print_str_train + NEW_LINE)
              print (print_str_train)

            # Save the model checkpoint periodically.
            if step % save_step == 0 or (step + 1) == FLAGS.max_steps:
              val_acc_r, val_n_correct_r = sess.run([acc, n_correct], feed_dict=val_feed)

              frmt_str = ' step %d, Val Acc = %.2f, num correct = %d'
              print_str_val = frmt_str % (step, val_acc_r, val_n_correct_r)
              val_f.write(print_str_val + NEW_LINE)
              print(print_str_val)

              checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=step)


      except tf.errors.OutOfRangeError:
        print ('Done training -- epoch limit reached')

      finally:
        # When done, ask the threads to stop.
        val_f.write(NEW_LINE +
                    NEW_LINE +
                    '############################ FINISHED ############################' +
                    NEW_LINE)
        val_f.close()
        coord.request_stop()

      # Wait for threads to finish.
      coord.join(threads)
      sess.close()

    else:

      ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
      print ckpt.model_checkpoint_path
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restored!')

      for i in range(100):
        images_val_r, labels_val_r = sess.run([val_images, val_labels])
        val_feed = {images_placeholder: images_val_r,
                    labels_placeholder: labels_val_r}

        tf.scalar_summary('Acc', acc)

        print('Calculating Acc: ')

        acc_r = sess.run(acc, feed_dict=val_feed)
        print(acc_r)

    coord.join(threads)
    sess.close()


def main(argv=None):
  train()


if __name__ == '__main__':
  tf.app.run()
