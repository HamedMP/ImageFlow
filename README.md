# ImageFlow
A simple wrapper of TensorFlow for Image I/O in pure tensor flow format

You can read all PNG and JPG/JPEG images in a directory with TensorFlow buil-in functions to boost speed.
Supported formats by TensorFlow are: PNG, JPG/JPEG

Currently it reads all PNG/JPG/JPEG images in a directory, the way most of the machine learning problems organized.

Return: A Tensor (numpy array) of object of type uint8. 3-D with shape [height, width, channels]

Usage:
  Include the file in your project.
  ```python
    image_list = read_images(path_to_dir) # 'path to the directory of your training images, JPG/PNG only
  ```
If you want to visualize it:
  ```python
    PIL.Image.show(Image.fromarray(image_list[0])) # You can use any library to show it, PIL, CV2, ...
  ```
If you don't want to use additional file in your project and just want to know how to import an image with TensorFlow:
```python
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

Dependencies:

* TensorFlow
* Numpy