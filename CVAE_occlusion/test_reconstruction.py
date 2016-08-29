from configuration import *
from util import *
import numpy as np
import tensorflow as tf
from random import randint
from tensor_definition import test_epsilon, test_x_hat, test_y
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST')

saver = tf.train.Saver()

def test(sess):
    normalized_z = np.random.normal(0, 1.0, latent_dim).reshape((1, latent_dim))
    rand_idx = randint(0, mnist.test.num_examples)
    the_image = mnist.test.images[rand_idx]
    the_origin_image = np.copy(the_image)
    the_occ_image = randomOcclude(the_image.reshape((1, 784)))
    feed_dict = {test_epsilon: normalized_z, test_y: the_occ_image}
    r = sess.run(test_x_hat, feed_dict=feed_dict)
    return [r.reshape((28, 28)), the_origin_image.reshape((28,28)), the_occ_image.reshape((28,28))]


results = []
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(cvae_model_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, cvae_model_path + model_name)
        print("Model loaded")
        for i in range(test_iter):
            results.append(test(sess))
    else:
        print("No checkpoint file found")


# analyze the results
from matplotlib import pyplot as plt

def saveFig(img, path):
    plt.imsave(path, img)

for i in range(test_iter):
    r = results[i]
    saveFig(r[0], cvae_model_path + str(i) + '.reconstructed.png')
    saveFig(r[1], cvae_model_path + str(i) + '.origin.png')
    saveFig(r[2], cvae_model_path + str(i) + '.occluded.png')

print("Finished")
