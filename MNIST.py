import os
import time

import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def sparse_target(input_data):
    temp = [[0, 1] if i == 1 else [1, 0] for i in input_data]
    temp = pd.DataFrame(temp, columns=['a', 'b'])
    return temp


mnist = input_data.read_data_sets("MNIST", one_hot=True)

DTYPE = tf.float64
BATCH_SIZE = 100
TRAINING_ROUNDS = 30000
PRINT_STEPS = round(TRAINING_ROUNDS / 30, 0)
global_step = tf.Variable(0.0, trainable=False, dtype=DTYPE)

# Initialization Variable
weight1 = tf.Variable(tf.truncated_normal([784, 500], stddev=1, dtype=DTYPE))
weight2 = tf.Variable(tf.truncated_normal([500, 10], stddev=1, dtype=DTYPE))
biases1 = tf.Variable(tf.constant(0.1, shape=[500], dtype=DTYPE))
biases2 = tf.Variable(tf.constant(0.1, shape=[10], dtype=DTYPE))

x = tf.placeholder(DTYPE, shape=[None, 784], name="train_input")
y_ = tf.placeholder(DTYPE, shape=[None, 10], name="target_output")

xTest = tf.placeholder(DTYPE, shape=[None, 784], name="train_input")
yTest_ = tf.placeholder(DTYPE, shape=[None, 10], name="target_output")

# Moving Average
va = tf.train.ExponentialMovingAverage(0.99, global_step)
vao = va.apply(tf.trainable_variables())

layer1 = tf.nn.relu(tf.matmul(x, weight1) + biases1)
y = tf.nn.relu(tf.matmul(layer1, weight2) + biases2)

layer1_average = tf.nn.relu(tf.matmul(x, va.average(weight1)) + va.average(biases1))
y_average = tf.nn.relu(tf.matmul(layer1_average, va.average(weight2)) + va.average(biases2))

# layer11 = tf.nn.relu(tf.matmul(xTest, weight1) + biases1)
# yTest = tf.nn.relu(tf.matmul(layer11, weight2) + biases2)
layer11 = tf.nn.relu(tf.matmul(xTest, va.average(weight1)) + va.average(biases1))
yTest = tf.nn.relu(tf.matmul(layer11, va.average(weight2)) + va.average(biases2))

# Define Loss Function
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
loss = tf.reduce_mean(cross_entropy)

# Training Function
learn_rate = tf.train.exponential_decay(0.8, global_step, mnist.train.num_examples / BATCH_SIZE, 0.9, staircase=True)
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, global_step=global_step)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
with tf.control_dependencies([train_step, vao]):
    train_op = tf.no_op(name='train')

# Accuracy Function
prediction = tf.equal(tf.argmax(yTest_, 1), tf.argmax(yTest, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, DTYPE))

for j in range(0, 1):
    # record starting time
    time_start = time.time()

    print("------{}-------".format(j))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_feed = {xTest: mnist.test.images, yTest_: mnist.test.labels}

        print("time:{}, Rounds:{}, Accuracy: {}".format(round(time.time() - time_start, 0), 0,
                                                        sess.run(accuracy, feed_dict=test_feed)))

        for i in range(1, TRAINING_ROUNDS):
            xtrain, ytrain = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xtrain, y_: ytrain})

            if i % PRINT_STEPS == 0:
                print("time:{}, Rounds:{}, Accuracy: {}".format(round(time.time() - time_start, 0), i,
                                                                sess.run(accuracy, feed_dict=test_feed)))

        print("time:{}, Rounds:{}, Accuracy: {}".format(round(time.time() - time_start, 0), "Final",
                                                        sess.run(accuracy, feed_dict=test_feed)))
        print(sess.run(tf.argmax(yTest, 1), feed_dict=test_feed))
        print(sess.run(tf.argmax(yTest_, 1), feed_dict=test_feed))
