import tensorflow as tf
DTYPE = tf.float64

a = tf.placeholder(DTYPE, shape=[None, 3], name="a_input")
aa = [[1,2,3]]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

sess.run(a, feed_dict={a:aa})

# print(sess.run(a))
