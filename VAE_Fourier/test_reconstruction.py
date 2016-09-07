from configuration import *
import numpy as np
import tensorflow as tf
from util import *
from os.path import realpath
from random import randint
from tensor_definition import x, x_hat_output
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST')

saver = tf.train.Saver()

def test(sess):
    rand_idx = randint(0, mnist.test.num_examples)
    the_image = mnist.test.images[rand_idx]
    the_origin_image = np.copy(the_image)
    occ = randomOcclude(the_image.reshape((1, input_dim)))
    feed_dict = {x: occ}
    r = sess.run(x_hat_output, feed_dict=feed_dict)
    return [r.reshape((28,28)), the_origin_image.reshape((28, 28)), occ.reshape((28,28))]


results = []
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(vae_model_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, vae_model_path + model_name)
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
    saveFig(r[0], vae_model_path + str(i) + '.reconstructed.png')
    saveFig(r[1], vae_model_path + str(i) + '.origin.png')
    saveFig(r[2], vae_model_path + str(i) + '.occ.png')

print("Finished")


# The optional feed_dict argument allows the caller to override the value of tensors in the graph.
# Each key in feed_dict can be one of the following types:
#
# If the key is a Tensor, the value may be a Python scalar, string, list, or numpy ndarray
# that can be converted to the same dtype as that tensor. Additionally, if the key is a placeholder,
# the shape of the value will be checked for compatibility with the placeholder.
#
# If the key is a SparseTensor, the value should be a SparseTensorValue.
# If the key is a nested tuple of Tensors or SparseTensors, the value should be a nested tuple with the same
#  structure that maps to their corresponding values as above.
#
# Each value in feed_dict must be convertible to a numpy array of the dtype of the corresponding key.