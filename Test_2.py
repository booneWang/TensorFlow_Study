import tensorflow as tf

a = tf.Variable(tf.constant(0.1, shape=[10], dtype=tf.float64))

with tf.Session() as sess:
    ini = tf.global_variables_initializer()
    sess.run(ini)
    print(sess.run(a))