import tensorflow as tf
from Titanic import *
import pandas as pd


def sparse_target(input_data):
    temp = [[0, 1] if i == 1 else [1, 0] for i in input_data]
    temp = pd.DataFrame(temp, columns=['a', 'b'])
    return temp


DTYPE = tf.float64
BATCH_SIZE = 100
TRAINING_ROUNDS = 5000
PRINT_STEPS = round(TRAINING_ROUNDS / 5, 0)

x_train, x_test, y_train, y_test = get_titanic()
y_train = sparse_target(y_train)
y_test = sparse_target(y_test)

# Initialization Variable
weight1 = tf.Variable(tf.random_normal([13, 20], stddev=1, dtype=DTYPE))
weight2 = tf.Variable(tf.random_normal([20, 2], stddev=1, dtype=DTYPE))
biases1 = tf.Variable(tf.constant(1.0, shape=[20], dtype=DTYPE))
biases2 = tf.Variable(tf.constant(1.0, shape=[2], dtype=DTYPE))

x = tf.placeholder(DTYPE, shape=[None, 13], name="train_input")
y_ = tf.placeholder(DTYPE, shape=[None, 2], name="target_output")
layer1 = tf.nn.relu(tf.matmul(x, weight1) + biases1)
y = tf.nn.relu(tf.matmul(layer1, weight2) + biases2)

xTest = tf.placeholder(DTYPE, shape=[None, 13], name="train_input")
yTest_ = tf.placeholder(DTYPE, shape=[None, 2], name="target_output")
layer11 = tf.nn.relu(tf.matmul(xTest, weight1) + biases1)
yTest = tf.nn.relu(tf.matmul(layer11, weight2) + biases2)

# Define Loss Function
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
loss = tf.reduce_mean(cross_entropy)

# Training Function
global_step = tf.Variable(0.0, trainable=False, dtype=DTYPE)
learn_rate = tf.train.exponential_decay(0.8, global_step, 100, 0.9, staircase=True)
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, global_step=global_step)
# train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

# Accuracy Function
prediction = tf.equal(tf.argmax(yTest_, 1), tf.argmax(yTest, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, DTYPE))

for j in range(0, 10):
    print("------{}-------".format(j))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_feed = {x: x_train, y_: y_train}
        test_feed = {xTest: x_test, yTest_: y_test}

        print("Rounds:{}, Accuracy: {}".format(0, sess.run(accuracy, feed_dict=test_feed)))

        for i in range(1, TRAINING_ROUNDS):
            sess.run(train_step, feed_dict=train_feed)

            if i % PRINT_STEPS == 0:
                print("Rounds:{}, Accuracy: {}".format(i, sess.run(accuracy, feed_dict=test_feed)))

        print("Rounds:{}, Accuracy: {}".format("Final", sess.run(accuracy, feed_dict=test_feed)))
        print(sess.run(tf.argmax(yTest, 1), feed_dict=test_feed))
        print(sess.run(tf.argmax(yTest_, 1), feed_dict=test_feed))
