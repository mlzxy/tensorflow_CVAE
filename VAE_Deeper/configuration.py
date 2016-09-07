input_dim = 3*32*32
hidden_encoder_dim = 800
hidden_decoder_dim = 800
latent_dim = 40
regularization_weight = 0
euclidean_weight = 0.0


# Iteration Number and batch size
n_steps = 1e5
batch_size = 1000
lr = 0.001

# Model save
vae_model_path = './save/'
model_name = 'model.ckpt'
vae_summary_name = 'vae_train'
snapshot_on = 50

# test
test_iter = 100


#
