import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

mnist = input_data.read_data_sets("mnist", one_hot=True)

image_size = 784
cond_size = 10
batch_size = 128
learning_rate = 0.001
num_iteration = 1000000
hidden_size = 10
z_size = 8
sample_num = 5

def encoder(x, cond):
    input_e = tf.concat([x, cond], axis=1)
    # first layer:
    w0_e = tf.Variable(tf.truncated_normal([image_size + cond_size, hidden_size], mean=0, stddev=0.1), dtype=tf.float32)
    b0_e = tf.Variable(tf.zeros(hidden_size), dtype=tf.float32)
    h1_e = tf.nn.relu(tf.matmul(input_e, w0_e) + b0_e)

    w_mean = tf.Variable(tf.truncated_normal([hidden_size, z_size], mean=0, stddev=0.1), dtype=tf.float32)
    b_mean = tf.Variable(tf.zeros(z_size), dtype=tf.float32)
    q_mean = tf.matmul(h1_e, w_mean) + b_mean

    w_sigma = tf.Variable(tf.truncated_normal([hidden_size, z_size], mean=0, stddev=0.1), dtype=tf.float32)
    b_sigma = tf.Variable(tf.zeros(z_size), dtype=tf.float32)
    q_sigma = tf.matmul(h1_e, w_sigma) + b_sigma
    return q_mean, q_sigma

def decoder(z, cond):
    input = tf.concat([z, cond], axis=1)
    h1 = tf.nn.relu(tf.matmul(input, w0) + b0)
    x_hat = tf.matmul(h1, w1) + b1
    return x_hat

# training placeholders:
x = tf.placeholder(tf.float32, [None, image_size], 'input')
cond = tf.placeholder(tf.float32, [None, 10], 'condition')
z = tf.placeholder(tf.float32, [None, z_size], 'latent_variable')

# decoder variables:
w0 = tf.Variable(tf.truncated_normal([z_size + cond_size, hidden_size], mean=0, stddev=0.1), dtype=tf.float32)
b0 = tf.Variable(tf.zeros(hidden_size), dtype=tf.float32)

w1 = tf.Variable(tf.truncated_normal([hidden_size, image_size], mean=0, stddev=0.1), dtype=tf.float32)
b1 = tf.Variable(tf.zeros(image_size), dtype=tf.float32)

''' Model the network '''
q_mean, q_sigma = encoder(x, cond)

# take a sample from N(0, 1) for our re-parametrization technique
sample = tf.random_normal(tf.shape(q_mean), mean=0, stddev=1, dtype=tf.float32)
z = sample * q_sigma + q_mean
x_hat = decoder(z, cond)

# ELBO = likelihood - KL divergence
likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_hat), 1)
DKL = 0.5 * tf.reduce_sum(1. + tf.log(tf.pow(q_sigma, 2)) - tf.pow(q_mean, 2) - tf.pow(q_sigma, 2), 1)

loss = tf.reduce_mean(likelihood - DKL)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# generated image for test:
generated = decoder(z, cond)

''' Start training '''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

for i in range(num_iteration):
    batch = mnist.train.next_batch(batch_size)
    _, report_loss = sess.run([train_op, loss], feed_dict={x: batch[0], cond:batch[1]})

    if i%2000 == 0:
        print ("Iteration: %i\tLoss: %f\n" %(i, report_loss))

        # show and save CVAE results
        for i in range(10):
            cond_test = np.zeros(10)
            cond_test[i] = 1
            for j in range(sample_num):
                cond_test = np.reshape(cond_test, (1, 10))
                z_test = np.reshape(np.random.randn(z_size), [1, z_size])

                res = sess.run(generated, feed_dict={z: z_test, cond:cond_test})
                img = Image.fromarray(np.reshape(res[0], (28, 28)) * 255).convert('RGB')
                img.save('results/CVAE/' + str(i) + '_' + str(j+1) + '.png')
                # img.show()
