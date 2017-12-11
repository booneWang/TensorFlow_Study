import tensorflow as tf
import pandas as pd
from Titanic import *
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = get_titanic()


def sparse_target(input_data):
    temp = [[0, 1] if i == 1 else [1, 0] for i in input_data]
    temp = pd.DataFrame(temp, columns=['a', 'b'])
    return temp


def inference(x, w_1, w_2, w_3, w_4):
    middle1 = tf.nn.relu(tf.matmul(x, w_1))
    middle2 = tf.nn.relu(tf.matmul(middle1, w_2))
    middle3 = tf.nn.relu(tf.matmul(middle2, w_3))
    return tf.nn.relu(tf.matmul(middle3, w_4))


# Define train Data
x = tf.Variable(x_train)
y_ = tf.Variable(tf.cast(sparse_target(y_train), tf.float64))

# Define Test Data
xx = tf.Variable(x_test)
yy_ = tf.Variable(tf.cast(sparse_target(y_test), tf.float64))

# Define Weight
w1 = tf.Variable(tf.random_normal([13, 50], stddev=1, dtype=tf.float64))
w2 = tf.Variable(tf.random_normal([50, 50], stddev=1, dtype=tf.float64))
w3 = tf.Variable(tf.random_normal([50, 50], stddev=1, dtype=tf.float64))
w4 = tf.Variable(tf.random_normal([50, 2], stddev=1, dtype=tf.float64))

# Define Output
y = inference(x, w1, w2, w3, w4)
yy = inference(xx, w1, w2, w3, w4)

# Define Loss
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# test accuracy
compare = tf.equal(tf.argmax(yy, 1), tf.argmax(yy_, 1))
accuracy = tf.round(tf.reduce_mean(tf.cast(compare, tf.float64)) * 10000) / 10000

with tf.Session() as sess:
    # initialization all variable
    init = tf.global_variables_initializer()
    sess.run(init)

    STEPS = 5000
    PRINT_STEPS = round(STEPS / 5, 0)

    print("Round:{}, Accuracy:{}".format(0, sess.run(accuracy)))

    # train model
    for i in range(1, STEPS):
        sess.run(train_step)

        if i % PRINT_STEPS == 0:
            print("Round:{}, Accuracy:{}".format(i, sess.run(accuracy)))

    print("Round:{}, Accuracy:{}".format("final", sess.run(accuracy)))
    # print(sess.run(tf.argmax(yy, 1)))
    # print(sess.run(tf.argmax(yy_, 1)))
