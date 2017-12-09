import tensorflow as tf
DTYPE = tf.float64

def weight_structure(input_dim_num, output_dim_num, weight):
    layer_num = len(weight)+1
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

w,b = weight_structure(2,5, [11,500])

print(w)