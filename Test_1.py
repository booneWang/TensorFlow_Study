import tensorflow as tf
import pandas as pd
from Titanic import *
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = get_titanic()


def sparse_target(input_data):
    temp = [[0, 1] if i == 1 else [1, 0] for i in input_data]
    temp = pd.DataFrame(temp, columns=['a', 'b'])
    return temp


# Define train Data
x = tf.Variable(x_train)
y_ = tf.Variable(tf.cast(sparse_target(y_train), tf.float64))

# Define NN and weight
w1 = tf.Variable(tf.random_normal([13, 20], stddev=1, seed=1, dtype=tf.float64))
w2 = tf.Variable(tf.random_normal([20, 4], stddev=1, seed=1, dtype=tf.float64))
w3 = tf.Variable(tf.random_normal([4, 2], stddev=1, seed=1, dtype=tf.float64))

middle1 = tf.nn.relu(tf.matmul(x, w1))
middle2 = tf.nn.relu(tf.matmul(middle1, w2))
y = tf.nn.relu(tf.matmul(middle2, w3))

# Define Loss
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    # initialization all variable
    init = tf.global_variables_initializer()
    sess.run(init)

    STEPS = 5000

    print(sess.run(w3))

    # train model
    for i in range(1, STEPS):
        sess.run(train_step)

    print(sess.run(w3))
    # print(sess.run(w2))
