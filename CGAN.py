import tensorflow as tf

def generator(z, cond, hidden_size, output_size):
    input = tf.concat([z, cond], axis=1)

    # first layer:
    w0 = tf.Variable(tf.truncated_normal([int(input.get_shape()[1]), hidden_size], mean=0, stddev=0.1), dtype=tf.float32)
    b0 = tf.Variable(tf.zeros(hidden_size), dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(input, w0) + b0)

    w1 = tf.Variable(tf.truncated_normal([hidden_size, output_size], mean=0, stddev=0.1), dtype=tf.float32)
    b1 = tf.Variable(tf.zeros(output_size), dtype=tf.float32)
    x_hat = tf.nn.sigmoid(tf.matmul(h1, w1) + b1)

    return x_hat

def discriminator(x, cond, hidden_size):
    input = tf.concat([x, cond], axis=1)

    # first layer:
    w0 = tf.Variable(tf.truncated_normal([int(input.get_shape()[1]), hidden_size], mean=0, stddev=0.1), dtype=tf.float32)
    b0 = tf.Variable(tf.zeros(hidden_size), dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(input, w0) + b0)

    w1 = tf.Variable(tf.truncated_normal([hidden_size, 1], mean=0, stddev=0.1), dtype=tf.float32)
    b1 = tf.Variable(tf.zeros(1), dtype=tf.float32)
    # out = tf.nn.sigmoid(tf.matmul(h1, w1) + b1)
    out = tf.matmul(h1, w1) + b1

    return out

