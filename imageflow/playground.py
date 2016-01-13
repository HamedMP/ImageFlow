__author__ = 'HANEL'

import os
import glob
import tensorflow as tf
from PIL import Image
import numpy as np

import imageflow

imageflow.read_images()

'''
  Simple library to read all PNG and JPG/JPEG images in a directory
  with TensorFlow buil-in functions to boost speed.

  Hamed MP
  Github: @hamedmp
  Twitter: @hamedpc2002

'''


# def _read_png(filename_queue, num):
#
#   images = []
#   # filename_queue = tf.train.string_input_producer(filename_queue_list)
#   reader = tf.WholeFileReader()
#   key, value = reader.read(filename_queue)
#
#   _img = tf.image.decode_png(value)
#
#   init_op = tf.initialize_all_variables()
#   with tf.Session() as sess:
#     sess.run(init_op)
#
#     # Start populating the filename queue.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(1):
#       png = _img.eval()
#       images.append(png)
#       Image._showxv(Image.fromarray(np.asarray(png)))
#
#     coord.request_stop()
#     coord.join(threads)
#
#
#   return images
#
#
# def _read_jpg(filename_queue, num):
#
#   images = []
#   # filename_queue = tf.train.string_input_producer(filename_queue_list)
#   reader = tf.WholeFileReader()
#   key, value = reader.read(filename_queue)
#
#   _img = tf.image.decode_jpeg(value)
#
#   init_op = tf.initialize_all_variables()
#   with tf.Session() as sess:
#     sess.run(init_op)
#
#     # Start populating the filename queue.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(1):
#       jpeg = _img.eval()
#       images.append(jpeg)
#       Image._showxv(Image.fromarray(np.asarray(jpeg)))
#
#     coord.request_stop()
#     coord.join(threads)
#
#
#   return images
  #
  #   print(jpeg.shape)


def read_images(path, is_directory=True):

  images = []
  png_files = []
  jpeg_files = []

  reader = tf.WholeFileReader()

  png_files_path = glob.glob(os.path.join(path, '*.[pP][nN][gG]'))
  jpeg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][eE][gG]')) #,
  jpg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][gG]')) # glob.glob(os.path.join(path + '*.[jJ][pP][nN][gG]'))

  print(png_files_path)
  # jpeg_files_path = [glob.glob(path + '*.jpg'), glob.glob(path + '*.jpeg')]

  if is_directory:
    for filename in png_files_path:
      png_files.append(filename)
    for filename in jpeg_files_path:
      jpeg_files.append(filename)
    for filename in jpg_files_path:
      jpeg_files.append(filename)
  else:
     _, extension = os.path.splitext(path)
     print(extension)
     if extension.lower() == '.png':
      key, value = reader.read(tf.train.string_input_producer(path))
      img = tf.image.decode_png(value)
      print(img)
      Image._show(Image.fromarray(np.asarray(img)))
      return img


  # Decode if there is a PNG file:
  if len(png_files) > 0:
    png_file_queue = tf.train.string_input_producer(png_files)
    pkey, pvalue = reader.read(png_file_queue)
    p_img = tf.image.decode_png(pvalue)

  if len(jpeg_files) > 0:
    jpeg_file_queue = tf.train.string_input_producer(jpeg_files)
    jkey, jvalue = reader.read(jpeg_file_queue)
    j_img = tf.image.decode_jpeg(jvalue)


  with tf.Session() as sess:

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    if len(png_files) > 0:
      for i in range(len(png_files)):
        png = p_img.eval()
        images.append(png)

      # Image._showxv(Image.fromarray(np.asarray(png)))

    if len(jpeg_files) > 0:
      for i in range(len(jpeg_files)):
        jpeg = j_img.eval()
        images.append(jpeg)

    coord.request_stop()
    coord.join(threads)

  return images

  # all_images = [_read_png(png_file_queue, num=len(png_files)),
  #               _read_jpg(jpeg_file_queue, num=len(jpeg_files))]

  # return all_images


read_images('/Users/HANEL/Desktop/')


# TODO: Remove

def _read_jpg():


  dumm = glob.glob('/Users/HANEL/Desktop/' + '*.png')
  print(len(dumm))
  filename_queue = tf.train.string_input_producer(dumm)
  # filename_queue = tf.train.string_input_producer(['/Users/HANEL/Desktop/tf.png', '/Users/HANEL/Desktop/ft.png'])

  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)

  my_img = tf.image.decode_png(value)
  # my_img_flip = tf.image.flip_up_down(my_img)

  init_op = tf.initialize_all_variables()
  with tf.Session() as sess:
    sess.run(init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1):
      gunel = my_img.eval()

      print(gunel.shape)

    Image._showxv(Image.fromarray(np.asarray(gunel)))
    coord.request_stop()
    coord.join(threads)

#
# _read_jpg()
