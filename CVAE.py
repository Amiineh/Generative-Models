import tensorflow as tf
import numpy as np


def encoder(x, cond, hidden_size, z_size):
    ''' This function takes the input x and returns mean and stddev of q(z|x) '''
    # concatenate input and condition:
    input = tf.concat([x, cond], axis=1)

    # first layer:
    w0 = tf.Variable(tf.truncated_normal([int(input.get_shape()[1]), hidden_size], mean=0, stddev=0.1), dtype=tf.float32)
    b0 = tf.Variable(tf.zeros(hidden_size), dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(input, w0) + b0)

    w_mean = tf.Variable(tf.truncated_normal([hidden_size, z_size], mean=0, stddev=0.1), dtype=tf.float32)
    b_mean = tf.Variable(tf.zeros(z_size), dtype=tf.float32)
    mean = tf.matmul(h1, w_mean) + b_mean

    w_sigma = tf.Variable(tf.truncated_normal([hidden_size, z_size], mean=0, stddev=0.1), dtype=tf.float32)
    b_sigma = tf.Variable(tf.zeros(z_size), dtype=tf.float32)
    sigma = tf.matmul(h1, w_sigma) + b_sigma

    return mean, sigma


def decoder(z, cond, hidden_size, output_size):
    ''' This function takes the encoded tensor z, and reconstructs y '''
    # concatenate input and condition:
    input = tf.concat([z, cond], axis=1)

    # first layer:
    w0 = tf.Variable(tf.truncated_normal([int(input.get_shape()[1]), hidden_size], mean=0, stddev=0.1), dtype=tf.float32)
    b0 = tf.Variable(tf.zeros(hidden_size), dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(input, w0) + b0)

    w1 = tf.Variable(tf.truncated_normal([hidden_size, output_size], mean=0, stddev=0.1), dtype=tf.float32)
    b1 = tf.Variable(tf.zeros(output_size), dtype=tf.float32)
    x_hat = tf.matmul(h1, w1) + b1

    return x_hat

def autoEncode(x, cond, hidden_size, z_size):
    ''' This function uses the encoder and encoder to reconstruct images '''
    q_mean, q_sigma = encoder(x, cond, hidden_size, z_size)

    # take a sample from N(0, 1) for our re-parametrization technique
    sample = tf.random_normal(tf.shape(q_mean), mean=0, stddev=1, dtype=tf.float32)
    z = sample * q_sigma + q_mean

    x_hat = decoder(z, cond, hidden_size, int(x.get_shape()[1]))

    # ELBO = likelihood - KL divergence
    likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_hat), 1)
    DKL = 0.5 * tf.reduce_sum(1. + tf.log(tf.pow(q_sigma, 2)) - tf.pow(q_mean, 2) - tf.pow(q_sigma, 2), 1)

    loss = tf.reduce_mean(likelihood - DKL)

    return x_hat, loss, q_mean, q_sigma


