import tensorflow as tf
from configuration import *
import tensorflow.contrib.layers as layers


x = tf.placeholder("float", shape=[None, SIZE, SIZE, CHANNEL])


# Encoder #

## conv layer 1
conv_1_x = layers.convolution2d(x, kernel_size=[5,5], num_outputs=64, padding='SAME', activation_fn=tf.nn.relu,
                                normalizer_fn=layers.batch_norm)


## conv layer 2.res1
conv_2_x_r1 = layers.convolution2d(conv_1_x, kernel_size=[3,3], num_outputs=64, padding='SAME', activation_fn=tf.nn.relu,
                                normalizer_fn=layers.batch_norm) + conv_1_x

## conv layer 2.res2
conv_2_x_r2 = layers.convolution2d(conv_2_x_r1, kernel_size=[3,3], num_outputs=64, padding='SAME', activation_fn=tf.nn.relu,
                                normalizer_fn=layers.batch_norm) + conv_2_x_r1

## conv layer 2.res3
conv_2_x_r3 = layers.convolution2d(conv_2_x_r2, kernel_size=[3,3], num_outputs=64, padding='SAME', activation_fn=tf.nn.relu,
                                normalizer_fn=layers.batch_norm) + conv_2_x_r2

## conv layer 3
conv_3_x = layers.convolution2d(conv_2_x_r3, kernel_size=[5,5], num_outputs=16, padding='SAME', activation_fn=tf.nn.relu,
                                normalizer_fn=layers.batch_norm)

## conv layer 3.res1
conv_3_x_r1 = layers.convolution2d(conv_3_x, kernel_size=[5,5], num_outputs=16, padding='SAME', activation_fn=tf.nn.relu,
                                normalizer_fn=layers.batch_norm) + conv_3_x

## conv layer 4
mu_encoder = layers.convolution2d(conv_3_x_r1, kernel_size=[3,3], num_outputs=4, padding='SAME', activation_fn=tf.nn.relu)
sigma_encoder = layers.convolution2d(conv_3_x_r1, kernel_size=[3,3], num_outputs=4, padding='SAME', activation_fn=tf.nn.relu)



# Z #
epsilon = tf.random_normal(tf.shape(mu_encoder), name='epsilon')
std_encoder = tf.exp(0.5 * sigma_encoder)
z = mu_encoder + tf.mul(epsilon, std_encoder)


# Decoder #
## conv layer 1
conv_1_z = layers.convolution2d(z, kernel_size=[3,3], num_outputs=64, padding='SAME', activation_fn=tf.nn.relu,
                                normalizer_fn=layers.batch_norm)

## conv layer 1.res1
conv_1_z_r1 = layers.convolution2d(conv_1_z, kernel_size=[3,3], num_outputs=64, padding='SAME', activation_fn=tf.nn.relu,
                                normalizer_fn=layers.batch_norm) + conv_1_z

## conv layer 1.res2
conv_1_z_r2 = layers.convolution2d(conv_1_z_r1, kernel_size=[3,3], num_outputs=64, padding='SAME', activation_fn=tf.nn.relu,
                                normalizer_fn=layers.batch_norm) + conv_1_z_r1

## conv layer 1.res3
conv_1_z_r3 = layers.convolution2d(conv_1_z_r2, kernel_size=[3,3], num_outputs=64, padding='SAME', activation_fn=tf.nn.relu,
                                normalizer_fn=layers.batch_norm) + conv_1_z_r2

## conv layer 2
conv_2_z = layers.convolution2d(conv_1_z_r3, kernel_size=[5,5], num_outputs=16, padding='SAME', activation_fn=tf.nn.relu,
                                normalizer_fn=layers.batch_norm)

## conv layer 2.res1
conv_2_z_r1 = layers.convolution2d(conv_2_z, kernel_size=[5,5], num_outputs=16, padding='SAME', activation_fn=tf.nn.relu,
                                normalizer_fn=layers.batch_norm) + conv_2_z


## conv layer 3
conv_3_z = layers.convolution2d(conv_2_z_r1, kernel_size=[5,5], num_outputs=4, padding='SAME', activation_fn=tf.nn.relu,
                                normalizer_fn=layers.batch_norm)

## conv layer 4
conv_4_z = layers.convolution2d(conv_3_z, kernel_size=[3,3], num_outputs=CHANNEL, padding='SAME', activation_fn=None)
x_hat_non_activation = conv_4_z
x_hat = tf.nn.sigmoid(x_hat_non_activation)


BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_hat_non_activation, x), reduction_indices=1)
KLD = -0.5 * tf.reduce_sum(1 + sigma_encoder - tf.pow(mu_encoder, 2) - tf.exp(sigma_encoder), reduction_indices=1)
loss = tf.reduce_mean(BCE + KLD)


# Train #
train_step = tf.train.AdamOptimizer(lr).minimize(loss)


# Summary #
loss_summary = tf.scalar_summary("loss", loss)
summary_op = tf.merge_all_summaries()


