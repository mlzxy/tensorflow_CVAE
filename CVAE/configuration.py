DataName = 'MNIST'
Adam = 'Adam'
RMSProp = 'RMSProp'


input_dim = 784
hidden_encoder_dim = 400
hidden_decoder_dim = 400

latent_dim = 20
y_dim_first = 10
y_dim_second = 10
latent_y_dim_first = 5
latent_y_dim_second = 5

regularization_weight = 0
euclidean_weight = 1.0

# Iteration Number and batch size
n_steps = 5000
batch_size = 600
lr = 0.01
momentum = 0.6
decay = 0.8
optimizer = RMSProp


# Model save
cvae_model_path = './save/'
model_name = 'model.ckpt'
cvae_summary_name = 'cvae_train'
snapshot_on = 50


# test
test_iter = 100