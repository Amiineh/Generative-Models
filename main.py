import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import CVAE

mnist = input_data.read_data_sets("mnist", one_hot=True)

image_size = 784
batch_size = 128
learning_rate = 0.001
num_iteration = 1000000
hidden_size = 10
z_size = 8
sample_num = 5

# training data:
x = tf.placeholder(tf.float32, [None, image_size], 'input')
cond = tf.placeholder(tf.float32, [None, 10], 'condition')

x_hat, loss = CVAE.autoEncode(x, cond, hidden_size, z_size)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# test data:
z = tf.placeholder(tf.float32, [None, z_size], 'latent')
fake_cond = tf.placeholder(tf.float32, [None, 10], 'test_condition')

sample = tf.sigmoid(CVAE.decoder(z, fake_cond, hidden_size, image_size))

# Start training:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# CVAE training:
sess.run(tf.global_variables_initializer())

for i in range(num_iteration):
    batch = mnist.train.next_batch(batch_size)
    _, report_loss, res = sess.run([train_op, loss, x_hat], feed_dict={x: batch[0], cond:batch[1]})

    if i%3000 == 0:
        print ("Iteration: %i\tLoss: %f\n" %(i, report_loss))

        # y = np.zeros(shape=[16, 10])
        # y[:, np.random.randint(0, 10)] = 1.
        #
        # samples = sess.run(sample, feed_dict={z: np.random.randn(16, z_size), fake_cond: y})
        # img = Image.fromarray(np.reshape(samples[0], (28, 28)) * 255).convert('RGB')
        # # img.save('results/CVAE/' + str(i) + '_' + str(j+1) + '.png')
        # img.show()
        #
        # img2 = Image.fromarray(np.reshape(res[1], (28,28)) * 255).convert('RGB')
        # img2.show()

# show and save CVAE results
        for i in range(10):
            cond_test = np.zeros(10)
            cond_test[i] = 1
            for j in range(sample_num):
                z_test = np.random.randn(1, z_size)
                cond_test = np.reshape(cond_test, (1, 10))

                res = sess.run(sample, feed_dict={z:z_test, fake_cond:cond_test})
                img = Image.fromarray(np.reshape(res[0], (28, 28)) * 255).convert('RGB')
                img.save('results/CVAE/' + str(i) + '_' + str(j+1) + '.png')
                # img.show()
