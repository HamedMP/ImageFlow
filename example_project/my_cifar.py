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

"""The network model"""

__author__ = 'HANEL'

import tensorflow as tf

# Data
Data_PATH = '../../mcifar_data/'


# Network Parameters
n_input = 32 * 32 * 3  # Cifar ckpt input (img shape: 32*32)

out_conv_1 = 64
out_conv_2 = 64

n_hidden_1 = 384
n_hidden_2 = 192

dropout = 0.90  # Dropout, probability to keep units

# Global constants describing the CIFAR-10
NUM_CLASSES = 10  # Cifar10 total classes (0-9 digits)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 40000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Constants describing the training process.
NUM_EPOCHS_PER_DECAY = 10.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.60  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001  # Initial learning rate.

FLAGS = tf.app.flags.FLAGS


# Create model
def conv2d(img, w, b):
  return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b))


def max_pool(img, k):
  return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def inference(images):
  """Build the CIFAR model up to where it may be used for inference.
  Args:

  Returns:
    logits: Output tensor with the   computed logits.
  """

  # Reshape input picture
  print('In Inference ', images.get_shape(), type(images))

  images = tf.reshape(images, shape=[-1, 32, 32, 3])

  _dropout = tf.Variable(dropout)  # dropout (keep probability)

  # Store layers weight & bias
  _weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, out_conv_1], stddev=1e-3)),  # 5x5 conv, 3 input, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, out_conv_1, out_conv_2], stddev=1e-3)),
  # 5x5 conv, 64 inputs, 64 outputs
    'wd1': tf.Variable(tf.random_normal([out_conv_2 * 8 * 8, n_hidden_1], stddev=1e-3)),
    'wd2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=1e-3)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, NUM_CLASSES], stddev=1e-3))
  }

  _biases = {
    'bc1': tf.Variable(tf.random_normal([out_conv_1])),
    'bc2': tf.Variable(tf.random_normal([out_conv_2])),
    'bd1': tf.Variable(tf.random_normal([n_hidden_1])),
    'bd2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([NUM_CLASSES]))
  }

  # Convolution Layer 1
  with tf.name_scope('Conv1'):
    conv1 = conv2d(images, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k=2)
    # norm1
    conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)

  # Convolution Layer 2
  with tf.name_scope('Conv2'):
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    # norm2
    conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout)

  # Fully connected layer 1
  with tf.name_scope('Dense1'):
    dense1 = tf.reshape(conv2,
                        [-1, _weights['wd1'].get_shape().as_list()[0]])  # Reshape conv2 output to fit dense layer input
    dense1 = tf.nn.relu_layer(dense1, _weights['wd1'], _biases['bd1'])  # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout

  # Fully connected layer 2
  with tf.name_scope('Dense2'):
    dense2 = tf.nn.relu_layer(dense1, _weights['wd2'], _biases['bd2'])  # Relu activation

  # Output, class prediction
  logits = tf.add(tf.matmul(dense2, _weights['out']), _biases['out'])

  return logits


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for for "Loss" and "Loss/avg".

  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Reshape the labels into a dense Tensor of
  # shape [batch_size, NUM_CLASSES].
  sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
  indices = tf.reshape(tf.range(0, FLAGS.batch_size), [FLAGS.batch_size, 1])
  concated = tf.concat(1, [indices, sparse_labels])
  dense_labels = tf.sparse_to_dense(concated,
                                    [FLAGS.batch_size, NUM_CLASSES],
                                    1.0, 0.0)

  # Calculate the average cross entropy loss across the batch.
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits, dense_labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def training(loss, global_step):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  print('Decay steps is: ', decay_steps)
  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)

  # Create the adam or gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdamOptimizer(lr)
  # optimizer = tf.train.GradientDescentOptimizer(lr)

  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)

  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  print('Evaluation..')
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label's is was in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  num_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

  acc_percent = num_correct / FLAGS.batch_size

  # Return the number of true entries.
  return acc_percent * 100.0, num_correct


def main(argv=None):
  return 0


if __name__ == '__main__':
  tf.app.run()
