import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import CVAE
import CGAN

mnist = input_data.read_data_sets("mnist", one_hot=True)

# Hyper-parameters:
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

# test data:
z = tf.placeholder(tf.float32, [None, z_size], 'sample')
test_cond = tf.placeholder(tf.float32, [None, 10], 'test_condition')

# Session:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

''' ================================================ CVAE ================================================ '''
x_hat, loss, _, _ = CVAE.autoEncode(x, cond, hidden_size, z_size)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# epsilon = q_mean + q_sigma * tf.random_normal(tf.shape(q_mean), mean=0, stddev=1, dtype=tf.float32)
generated = CVAE.decoder(z, test_cond, hidden_size, image_size)

''' CVAE training '''
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

                res = sess.run(generated, feed_dict={z: z_test, test_cond:cond_test})
                img = Image.fromarray(np.reshape(res[0], (28, 28)) * 255).convert('RGB')
                img.save('results/CVAE/' + str(i) + '_' + str(j+1) + '.png')
                # img.show()



# ''' ================================================ CGAN ================================================ '''
#
# D_real = CGAN.discriminator(x, cond, hidden_size)
# G = CGAN.generator(z, cond, hidden_size, image_size)
# D_fake = CGAN.discriminator(G, cond, hidden_size)
#
# # loss of D is sum of evaluation of real images as 1 and fake images as 0
# loss_d_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real))
# loss_d_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake))
# loss_d = tf.reduce_mean(loss_d_real + loss_d_fake)
#
# # loss of G is how good it has fooled the discriminator, so evaluation of fake images as 1 by the discriminator
# loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
#
# train_op_d = tf.train.AdamOptimizer(learning_rate).minimize(loss_d)
# train_op_g = tf.train.AdamOptimizer(learning_rate).minimize(loss_g)
#
# # for test:
# generated = CGAN.generator(z, cond, hidden_size, image_size)
#
# ''' CGAN training '''
# sess.run(tf.global_variables_initializer())
#
# for i in range(num_iteration):
#     batch = mnist.train.next_batch(batch_size)
#     batch_z = np.random.rand(batch_size, z_size).astype(np.float32)
#
#     # update discriminator:
#     _, report_loss_d = sess.run([train_op_d, loss_d], feed_dict={x: batch[0], cond:batch[1], z: batch_z})
#
#     # update generator:
#     _, report_loss_g = sess.run([train_op_g, loss_g], feed_dict={cond: batch[1], z: batch_z})
#
#     if i%2000 == 0:
#         print ("Iteration: %i\t\tDiscriminator loss: %f\t\tGenerator loss: %f\n" %(i, report_loss_d, report_loss_g))
#
#         res = sess.run(generated, feed_dict={cond: batch[1], z: batch_z})
#         img = Image.fromarray(np.reshape(res[0], (28, 28)) * 255).convert('RGB')
#         img.show()
#
#         # show and save CGAN results
#         # for i in range(10):
#         #     cond_test = np.zeros(10)
#         #     cond_test[i] = 1
#         #     for j in range(sample_num):
#         #         cond_test = np.reshape(cond_test, (1, 10))
#         #         z_test = np.reshape(np.random.randn(z_size), [1, z_size])
#         #
#         #         res = sess.run(generated, feed_dict={z: z_test, test_cond:cond_test})
#         #         img = Image.fromarray(np.reshape(res[0], (28, 28)) * 255).convert('RGB')
#         #         img.save('results/CGAN/' + str(i) + '_' + str(j+1) + '.png')
#         #         # img.show()
#         #
#
