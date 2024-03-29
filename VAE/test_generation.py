from configuration import *
import numpy as np
import tensorflow as tf
from tensor_definition import test_epsilon, test_x_hat

saver = tf.train.Saver()

def test(sess):
    normalized_z = np.random.normal(0, 1.0, latent_dim).reshape((1, latent_dim))
    feed_dict = {test_epsilon: normalized_z}
    r = sess.run(test_x_hat, feed_dict=feed_dict)
    return [r.reshape((28, 28))]


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
    saveFig(r[0], vae_model_path + str(i) + '.generated_random.png')

print("Finished")
