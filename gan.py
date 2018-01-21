import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

def generator(z):
    h1_g = tf.nn.relu(tf.matmul(z, w0_g) + b0_g)
    G = tf.nn.sigmoid(tf.matmul(h1_g, w1_g) + b1_g)
    return G

def discriminator(x):
    h1_d = tf.nn.relu(tf.matmul(x, w0_d) + b0_d)
    D = tf.matmul(h1_d, w1_d) + b1_d
    return D

mnist = input_data.read_data_sets("mnist", one_hot=True)

# Hyper-parameters:
image_size = 784
label_size = 10
batch_size = 128
learning_rate = 0.001
num_iteration = 1000000
hidden_size = 128
z_size = 100
sample_num = 5

# training data:
x = tf.placeholder(tf.float32, [None, image_size], 'input')
z = tf.placeholder(tf.float32, [None, z_size], 'sample')

# discriminator variables:
w0_d = tf.Variable(tf.truncated_normal([image_size, hidden_size], mean=0, stddev=0.1), dtype=tf.float32)
b0_d = tf.Variable(tf.zeros(hidden_size), dtype=tf.float32)

w1_d = tf.Variable(tf.truncated_normal([hidden_size, 1], mean=0, stddev=0.1), dtype=tf.float32)
b1_d = tf.Variable(tf.zeros(1), dtype=tf.float32)

theta_d = [w0_d, b0_d, w1_d, b1_d]

# generator variables:
w0_g = tf.Variable(tf.truncated_normal([z_size, hidden_size], mean=0, stddev=0.1), dtype=tf.float32)
b0_g = tf.Variable(tf.zeros(hidden_size), dtype=tf.float32)

w1_g = tf.Variable(tf.truncated_normal([hidden_size, image_size], mean=0, stddev=0.1), dtype=tf.float32)
b1_g = tf.Variable(tf.zeros(image_size), dtype=tf.float32)

theta_g = [w0_g, b0_g, w1_g, b1_g]

# Session:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

D_real = discriminator(x)
G = generator(z)
D_fake = discriminator(G)

# loss of D is sum of evaluation of real images as 1 and fake images as 0
loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
loss_d = loss_d_real + loss_d_fake

# loss of G is how good it has fooled the discriminator, so evaluation of fake images as 1 by the discriminator
loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

train_op_d = tf.train.AdamOptimizer(learning_rate).minimize(loss_d, var_list=theta_d)
train_op_g = tf.train.AdamOptimizer(learning_rate).minimize(loss_g, var_list=theta_g)

''' CGAN training '''
sess.run(tf.global_variables_initializer())

for i in range(num_iteration):
    batch = mnist.train.next_batch(batch_size)
    batch_z = np.random.uniform(-1, 1, [batch_size, z_size]).astype(np.float32)

    # update discriminator:
    _, report_loss_d = sess.run([train_op_d, loss_d], feed_dict={z: batch_z, x: batch[0]})

    # update generator:
    _, report_loss_g = sess.run([train_op_g, loss_g], feed_dict={z: batch_z})

    if i%2000 == 0:
        print ("Iteration: %i\t\tDiscriminator loss: %f\t\tGenerator loss: %f\n" %(i, report_loss_d, report_loss_g))

        # show and save CGAN results
        for i in range(20):
            z_test = np.reshape(np.random.uniform(-1, 1, z_size), [1, z_size])

            res = sess.run(G, feed_dict={z: z_test})
            img = Image.fromarray(np.reshape(res[0], (28, 28)) * 255).convert('RGB')
            img.save('Results/GAN/sample ' + str(i) + '.png')
            # img.show()


