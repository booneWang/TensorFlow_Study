import tensorflow as tf
import matplotlib.pyplot as plt

# 初始化为0，后续会随迭代更新
gt = tf.Variable(0, trainable=False)
init = tf.global_variables_initializer()

# 迭代次数
rounds = 10000

# 输出初始化
x, y = [], []

with tf.Session() as sess:
    sess.run(init)
    learning_rate = tf.train.exponential_decay(0.8, gt, 30, 0.99)

    for i in range(0, rounds):
        x.append(i)
        y.append(sess.run(learning_rate, feed_dict={gt: i}))

    print(sess.run(gt))
plt.plot(x, y)
plt.show()
