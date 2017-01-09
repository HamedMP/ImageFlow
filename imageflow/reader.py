# Copyright 2016 Hamed MP. All Rights Reserved.
#
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

from __future__ import print_function

__author__ = 'HANEL'

import os
import glob
import csv
import tensorflow as tf
import numpy as np
from PIL import Image
from .utils import dense_to_one_hot


def _read_raw_images(path, is_directory=True):
  """Reads directory of images in tensorflow
  Args:
    path:
    is_directory:

  Returns:

  """
  images = []
  png_files = []
  jpeg_files = []

  reader = tf.WholeFileReader()

  png_files_path = glob.glob(os.path.join(path, '*.[pP][nN][gG]'))
  jpeg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][eE][gG]'))
  jpg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][gG]'))

  if is_directory:
    for filename in png_files_path:
      png_files.append(filename)
    for filename in jpeg_files_path:
      jpeg_files.append(filename)
    for filename in jpg_files_path:
      jpeg_files.append(filename)
  else:
    raise ValueError('Currently only batch read from directory supported')

  # Decode if there is a PNG file:
  if len(png_files) > 0:
    png_file_queue = tf.train.string_input_producer(png_files)
    pkey, pvalue = reader.read(png_file_queue)
    p_img = tf.image.decode_png(pvalue)

  if len(jpeg_files) > 0:
    jpeg_file_queue = tf.train.string_input_producer(jpeg_files)
    jkey, jvalue = reader.read(jpeg_file_queue)
    j_img = tf.image.decode_jpeg(jvalue)

  return  # TODO: return normal thing


def read_and_decode(filename_queue, imshape, normalize=False, flatten=True):
  """Reads
  Args:
    filename_queue:
    imshape:
    normalize:
    flatten:

  Returns:

  """
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64)
    })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.uint8)

  if flatten:
    num_elements = 1
    for i in imshape: num_elements = num_elements * i
    print(num_elements)
    image = tf.reshape(image, [num_elements])
    image.set_shape(num_elements)
  else:
    image = tf.reshape(image, imshape)
    image.set_shape(imshape)

  if normalize:
    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label


# Helper, Examples
def _read_labels_csv_from(path, num_classes, one_hot=False):
  """Reads
  Args:

  Returns:

  """
  print('Reading labels')
  with open(os.path.join(path), 'r') as dest_f:
    data_iter = csv.reader(dest_f)
    train_labels = [data for data in data_iter]

  train_labels = np.array(train_labels, dtype=np.uint32)

  if one_hot:
    labels_one_hot = dense_to_one_hot(train_labels, num_classes)
    labels_one_hot = np.asarray(labels_one_hot)
    return labels_one_hot

  return train_labels


def _read_pngs_from(path):
  """Reads directory of images.
  Args:
    path: path to the directory

  Returns:
    A list of all images in the directory in the TF format (You need to call sess.run() or .eval() to get the value).
  """
  images = []
  png_files_path = glob.glob(os.path.join(path, '*.[pP][nN][gG]'))
  for filename in png_files_path:
    im = Image.open(filename)
    im = np.asarray(im, np.uint8)

    # get only images name, not path
    image_name = filename.split('/')[-1].split('.')[0]
    images.append([int(image_name), im])

  images = sorted(images, key=lambda image: image[0])

  images_only = [np.asarray(image[1], np.uint8) for image in images]  # Use unint8 or you will be !!!
  images_only = np.array(images_only)

  print(images_only.shape)
  return images_only
