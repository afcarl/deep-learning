# mnist_mlp.py

import tensorflow as tf
from tensorflow.examples.tutorial.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSesseion()
