from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None
def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    #Set Parameters
    BATCH_SIZE = FLAGS.batch_size
    TRAIN_STEPS = FLAGS.train_steps

    n_pixels = 784
    learning_rate = 1e-3
    #Model Creation
    x = tf.placeholder(tf.float32, [None, n_pixels])
    dummy = tf.placeholder(tf.float32, [None, 1])

    W = tf.Variable(tf.zeros([1, n_pixels]))
    b = tf.Variable(tf.zeros([n_pixels]))
    y = tf.sigmoid(tf.matmul(dummy, W) + b)

    #Loss calculation
    cross_entropy = -1. * x * tf.log(y) - (1. - x) * tf.log(1. - y)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    #For plotting HeatMap
    plt.figure()
    plt.title('HeatMap MNIST')
    plt.text(0.2, 0.6,
        "Training started \n  Please Wait",
         transform=plt.gca().transAxes, fontsize=35)
    plt.ion()
    plt.show()


    with tf.Session() as sess:
        for digit in range(10):
            print("Generating heatmap for {}".format(digit))

            #initialization done for each digit to reset weights
            sess.run(init)

            for i_ in range(1, TRAIN_STEPS + 1):
                batch_xs = []

                #ugly hack to get only a batch of same digit
                while True:
                    t = mnist.train.next_batch(1)
                    if np.argmax(t[1][0]) == digit:
                        batch_xs.append(t[0][0])
                        if len(batch_xs) == BATCH_SIZE:
                            break

                y_out, _ = sess.run([y, train_step],
                    feed_dict={x: batch_xs,
                        dummy: [ [1] ]* BATCH_SIZE}) #dummy input of shape [BATCH_SIZE, 1]

                if i_ == TRAIN_STEPS:
                    #last iteration of a digit
                    #reshape 784 to (28, 28)
                    img = np.reshape(y_out[0], [28, 28])

                    #create a subplot for each digit
                    plt.subplot(2, 5, digit+1)
                    plt.imshow(img)
                    plt.draw()
                    plt.pause(0.01) #needed to update dynamically

    plt.ioff()
    plt.show() #for not closing the plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./mnist/input_data',
                      help='Directory for storing input data')
    parser.add_argument('--batch_size', type=int, default=50,
                      help='Batch size')
    parser.add_argument('--train_steps', type=int, default=500,
                      help='Training Steps')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
