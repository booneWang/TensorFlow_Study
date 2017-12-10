import tensorflow as tf
import pandas as pd
from Titanic import *
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST", one_hot=True)

# instant
DTYPE = tf.float64
#
STEPS = 100
PRINT_STEPS = round(STEPS / 5, 0)

INPUT_NODE_NUM = 784
OUTPUT_NODE_NUM = 10
LAYERS = [100, 100]


# make the Target sparse, like [0,1,0] to [[0,1],[1,0],[0,1]
def sparse_target(input_data):
    temp = [[0, 1] if i == 1 else [1, 0] for i in input_data]
    temp = pd.DataFrame(temp, columns=['a', 'b'])
    return temp


# Create the weight for NN
# input_dim_num - dimension # of x tensor (the train input)
# output_dim_num - dimension # of y tensor (the test output)
# weight - array, layers and elements of each layer
def weight_structure(input_dim_num, output_dim_num, weight):
    layer_num = len(weight) + 1
    weight_list = []
    biases = []
    start, end = 0, 0

    for j in range(0, layer_num):
        # first layer
        if j == 0:
            start = input_dim_num
            end = weight[0]

        # last layer
        elif j == layer_num - 1:
            start = end
            end = output_dim_num

        # middle layer
        else:
            start = end
            end = weight[j]

        w = tf.Variable(tf.random_normal([start, end], stddev=1, dtype=DTYPE))
        b = tf.Variable(tf.constant(0.1, shape=[end], dtype=DTYPE))

        weight_list.append(w)
        biases.append(b)
    return weight_list, biases


# Create the Neural Network
def nn_structure(input_tensor, weight_list, biases):
    layer_list = []
    for j in range(0, len(weight_list)):
        # when it's the first layer, then start from input tensor
        if j == 0:
            layer = tf.nn.relu(tf.matmul(input_tensor, weight_list[j]) + biases[j])
        # for other layer, start with previous layer's output
        else:
            layer = tf.nn.relu(tf.matmul(layer_list[j - 1], weight_list[j]) + biases[j])

        layer_list.append(layer)

    # return the last layer which is the output
    return layer_list[-1]


# Define train Data
x = tf.Variable(mnist.train.images, trainable=False, dtype=DTYPE)
y_ = tf.Variable(mnist.train.labels, trainable=False)

# Define Test Data
xx = tf.Variable(mnist.validation.images, trainable=False, dtype=DTYPE)
yy_ = tf.Variable(mnist.validation.labels, trainable=False)

weight_list, biases = weight_structure(INPUT_NODE_NUM, OUTPUT_NODE_NUM, LAYERS)
y = nn_structure(x, weight_list, biases)
yy = nn_structure(xx, weight_list, biases)

# Define Loss
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

#
Init_learning_rate = 0.8
global_step = tf.Variable(0.0, trainable=False, dtype=DTYPE)
decay_rate = 0.999
decay_steps = 0.0001
learn_rate = tf.train.exponential_decay(Init_learning_rate, global_step, decay_rate, decay_steps, staircase=True)

train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, global_step=global_step)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# test accuracy
compare = tf.equal(tf.argmax(yy, 1), tf.argmax(yy_, 1))
accuracy = tf.round(tf.reduce_mean(tf.cast(compare, DTYPE)) * 10000) / 10000

with tf.Session() as sess:
    # initialization all variable
    init = tf.global_variables_initializer()
    sess.run(init)

    print("Round:{}, Accuracy:{}".format(0, sess.run(accuracy)))

    # train model
    for i in range(1, STEPS):
        sess.run(train_step)

        if i % PRINT_STEPS == 0:
            print("Round:{}, Accuracy:{}".format(i, sess.run(accuracy)))

    print("Round:{}, Accuracy:{}".format("final", sess.run(accuracy)))
    # print(sess.run(tf.argmax(yy, 1)))
    # print(sess.run(tf.argmax(yy_, 1)))
