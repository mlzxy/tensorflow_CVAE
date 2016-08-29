from __future__ import division
from configuration import *
from util import *
import numpy as np
import os.path
import random
# Use Python3, python2 will probably cause numpy version problem with tf in osx.
# Because osx has a default python in /Library/System/..., To load tf python2.7 should change the sys.path first, then load.
import tensorflow as tf

# for debugging the occlusion part
# from matplotlib import pyplot as plt


# the mnist structure is like a class:  mnist.{validation,test,train}.{images,labels,num_examples, epochs_completed}
from tensorflow.examples.tutorials.mnist import input_data
import os
# Class DataSet: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
mnist = input_data.read_data_sets(DataName)
from tensor_definition import train_step, summary_op, loss, x, y


# add Saver ops, save training temp.
saver = tf.train.Saver()



# A class for running TensorFlow operations.
# A Session object encapsulates the environment in which Operation objects are executed,
# and Tensor objects are evaluated.
with tf.Session() as sess:
    # Writes Summary protocol buffers to event files.
    # The SummaryWriter class provides a mechanism to create an event file in a given directory
    # and add summaries and events to it. The class updates the file contents asynchronously.
    # This allows a training program to call methods to add data to the file directly from the training loop,
    # without slowing down training.
    if not os.path.exists(cvae_model_path):
        os.mkdir(cvae_model_path)
    summary_writer = tf.train.SummaryWriter(cvae_summary_name, # logdir, for visualizing training using summaries.
                                            graph_def=sess.graph_def)

    model_path = cvae_model_path + model_name
    # start training
    if os.path.isfile(model_path):
        print("Restoring saved parameters")
        saver.restore(sess, model_path)
    else:
        print("Initializing parameters")
        sess.run(tf.initialize_all_variables())

    for step in range(1, n_steps):
        # Very useful method from DataSet Class.
        batch, _ = mnist.train.next_batch(batch_size)

        feed_dict = {x: batch, y: randomOcclude(batch)}
        _, cur_loss, summary_str = sess.run([train_step, loss, summary_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        if step % snapshot_on == 0:
            save_path = saver.save(sess, model_path)
            print("Step {0} | Loss: {1}".format(step, cur_loss))
            if cur_loss < LOSS_LIMIT:
                print("Loss is small enough, exiting...")
                break




# The computations you'll use TensorFlow for - like training a massive deep neural network -
# can be complex and confusing. To make it easier to understand, debug, and optimize TensorFlow programs,
# we've included a suite of visualization tools called TensorBoard.
# tensorboard --logdir=./save/   // live preview the training process.