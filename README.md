# ImageFlow
A simple wrapper of TensorFlow for Converting, Importing (and Soon, Training) Images in tensorflow.

Installation:
```
pip install imageflow
```

Usage:

```python
import imageflow
```

#### Convert a directory of images and their labels to `.tfrecords`
Just calling the following function will make a `filename.tfrecords` file in the directory `converted_data` in your projects root(where you call this method).

```python
<<<<<<< Updated upstream
convert_images(images, labels, filename)
```

The `images` should be an array of shape `[-1, height, width, channel]` and has the same rows as the `labels`

#### Read distorted and normal data from `.tfrecords` in multi-thread manner:
```python
# Distorted images for training
images, labels = distorted_inputs(filename='../my_data_raw/train.tfrecords', batch_size=FLAGS.batch_size,
                                      num_epochs=FLAGS.num_epochs,
                                      num_threads=5, imshape=[32, 32, 3], imsize=32)

# Normal images for validation
val_images, val_labels = inputs(filename='../my_data_raw/validation.tfrecords', batch_size=FLAGS.batch_size,
                                    num_epochs=FLAGS.num_epochs,
                                    num_threads=5, imshape=[32, 32, 3])
```

=======
    filename_queue = tf.train.string_input_producer(['/Users/HANEL/Desktop/tf.png']) #  list of files to read
  
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
  
    my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.
  
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init_op)

    # Start populating the filename queue.
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1): #length of your filename list
      image = my_img.eval() #here is your image Tensor :) 

    print(image.shape)
    Image.show(Image.fromarray(np.asarray(image)))
    
    coord.request_stop()
    coord.join(threads)
```
>>>>>>> Stashed changes

Dependencies:

* TensorFlow
* Numpy
* Pillow
