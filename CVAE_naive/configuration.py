DataName = 'MNIST'
input_dim = 784
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 20
regularization_weight = 0
euclidean_weight = 1.0

# Iteration Number and batch size
n_steps = 5000
batch_size = 300
adam = 0.01


# Model save
cvae_model_path = './save/'
model_name = 'model.ckpt'
cvae_summary_name = 'cvae_train'
snapshot_on = 50


# test
test_iter = 10