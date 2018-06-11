#!/usr/bin/env python
import os
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
# read MNIST data set
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

# convolution
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# create symbolic variables
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# create variables: weights and biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define model

w1 = weight_variable([784,40])
b1 = weight_variable([40])
h1 = tf.nn.sigmoid(tf.matmul(X,w1)+b1)

w2 = weight_variable([40,10])
b2 = weight_variable([10])
h2 = tf.nn.sigmoid(tf.matmul(h1,w2)+b2)

y = tf.nn.softmax(h2)
# cross entropy
cross_entropy = -tf.reduce_sum(Y * tf.log(y)) +tf.nn.l2_loss(w1)*0.0001 +tf.nn.l2_loss(w2)*0.0001

# train step
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# init step
init = tf.initialize_all_variables()

with tf.Session() as sess:
	# run the init op
	sess.run(init)
	# then train
	for i in range(10000):
		if i%100 == 0:
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			print sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
		batch_trX, batch_trY = mnist.train.next_batch(128)
		sess.run(train_step, feed_dict={X: batch_trX, Y: batch_trY})

	# test and evaluate our model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
