import tensorflow as tf
from configuration import *

def weight_variable(shape):
    # Outputs random values from a truncated normal distribution
    # except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
    # hardware implementation is truncated normal for sure
    initial = tf.truncated_normal(shape, stddev=0.001)
    # tf.Variable is a class, When you train a model, you use variables to hold and update parameters.
    # Variables are in-memory buffers containing tensors. They must be explicitly initialized and can be saved to disk
    # during and after training. You can later restore saved values to exercise or analyse the model.
    # here is a matrix with shape. truncated normal is a initializer.
    return tf.Variable(initial)

def euclidean(x1, x2):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.sub(x1, x2)), reduction_indices=1))


def bias_variable(shape):
    # return a constant tensor, all 0.0 here
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)


# Inserts a placeholder for a tensor that will be always fed, usually the input_data
# one dimension here
x = tf.placeholder("float", shape=[None, input_dim])
# use for label
y = tf.placeholder("float", shape=[None, 10])

# regularization
l2_loss = tf.constant(0.0)

# the hidden dimension, the W/b are not used to calculate z, but the sufficient statistics of distribution of z,
# which are mu and sigma.
W_encoder_input_hidden = weight_variable([input_dim, hidden_encoder_dim])
b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)
hidden_encoder_without_y = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)


# Fuse the y information into the hidden_encoder.
W_encoder_y_hidden_first = weight_variable([10, hidden_encoder_dim])
b_encoder_y_hidden_first = bias_variable([hidden_encoder_dim])
y_first = tf.nn.relu(tf.matmul(y, W_encoder_y_hidden_first) + b_encoder_y_hidden_first)
l2_loss += tf.nn.l2_loss(W_encoder_y_hidden_first)
w_pixel_encoder = tf.Variable(0.5, name="weight_pixel_encoder")
w_label_encoder = tf.Variable(0.5, name="weight_label_encoder")
hidden_encoder = hidden_encoder_without_y*w_pixel_encoder + y_first*w_label_encoder


# Ok there is another layer between the mu and sigma, X->hidden->latent (mu/sigma)
# obviously we could use more
W_encoder_hidden_mu = weight_variable([hidden_encoder_dim, latent_dim])
b_encoder_hidden_mu = bias_variable([latent_dim])

# another regulariation
l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

# Mu encoder, output is mu
mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

# Seperate Layer for sigma
W_encoder_hidden_sigma = weight_variable([hidden_encoder_dim, latent_dim])
b_encoder_hidden_sigma = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_sigma)

# Sigma encoder
sigma_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_sigma) + b_encoder_hidden_sigma

# Sample epsilon, input for unprocessed z, could train truncated normal here to simulate hardware
epsilon = tf.random_normal(tf.shape(sigma_encoder), name='epsilon')

# Sample latent variable the exp part.
std_encoder = tf.exp(0.5 * sigma_encoder)

# z is the latent variable, now we go to the decoding part.
z = mu_encoder + tf.mul(std_encoder, epsilon)

###################### insert the y information into the latent variable.  #######################
W_decoder_y_to_latent = weight_variable([10, latent_dim])
l2_loss += tf.nn.l2_loss(W_decoder_y_to_latent)
b_decoder_y_to_latent = bias_variable([latent_dim])
y_decoder_first = tf.nn.relu(tf.matmul(y, W_decoder_y_to_latent) + b_decoder_y_to_latent)
w_z_decoder = tf.Variable(0.5, name="weight_z_decoder")
w_y_decoder = tf.Variable(0.5, name="weight_y_decoder")
zy = z * w_z_decoder + y_decoder_first * w_y_decoder
#########################################################

# Return from hidden, this is very similar to DRAW structure, but DRAW structure is less explainable in my opinion.
W_decoder_z_hidden = weight_variable([latent_dim , hidden_decoder_dim])
b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
# regularization all over the place.
l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

# Hidden layer decoder
hidden_decoder = tf.nn.relu(tf.matmul(zy, W_decoder_z_hidden) + b_decoder_z_hidden)

W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
b_decoder_hidden_reconstruction = bias_variable([input_dim])
l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)

# KL divergence.
# Computes the sum of elements across dimensions of a tensor.
KLD = -0.5 * tf.reduce_sum(1 + sigma_encoder - tf.pow(mu_encoder, 2) - tf.exp(sigma_encoder), reduction_indices=1)
# Reconstruction, will need to sigmoid it afterwards.
x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction


# Cross Entropy Loss
# Computes sigmoid cross entropy given logits.  -> (logits, targets, name=None)
# Measures the probability error in discrete classification tasks in which each class is
# independent and not mutually exclusive. For instance, one could perform multilabel
# classification where a picture can contain both an elephant and a dog at the same time.
# Ep(-log(q))
BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_hat, x), reduction_indices=1)

x_hat_output = tf.nn.sigmoid(x_hat)



# ===================================== Exportation for Training ===================================== #
# Computes the mean of elements across dimensions of a tensor.
loss = tf.reduce_mean(BCE + KLD)

# use euclidean loss?
loss = loss + euclidean_weight * tf.reduce_mean(euclidean(x, x_hat_output))

# OK, it is regularized_loss.
regularized_loss = loss + regularization_weight * l2_loss

# Outputs a Summary protocol buffer with scalar values.
loss_summ = tf.scalar_summary("loss", loss)
# Optimizer that implements the Adam algorithm, could try others.
if optimizer == Adam:
    train_step = tf.train.AdamOptimizer(lr).minimize(regularized_loss)
elif optimizer == RMSProp:
    train_step = tf.train.RMSPropOptimizer(lr, momentum=momentum, decay=decay).minimize(regularized_loss)
else:
    raise  "Not a valid optimizer"

# add op for merging summary, Merges all summaries collected in the default graph, like a common procedure.
summary_op = tf.merge_all_summaries()



# ===================================== Exportation for Inference ===================================== #
test_epsilon = tf.placeholder("float", shape=(None, latent_dim))
test_y = tf.placeholder("float", shape=(None, 10))
test_y_decoder_first = tf.nn.relu(tf.matmul(test_y, W_decoder_y_to_latent) + b_decoder_y_to_latent)
test_z = test_epsilon * w_z_decoder + test_y_decoder_first* w_y_decoder
test_hidden_decoder = tf.nn.relu(tf.matmul(test_z, W_decoder_z_hidden) + b_decoder_z_hidden)
test_x_hat = tf.nn.sigmoid(tf.matmul(test_hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction)