SIZE = 32
CHANNEL = 1

LABEL = 0

input_dim = CHANNEL*SIZE*SIZE
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 30
regularization_weight = 0.1
euclidean_weight = 0.0


# Iteration Number and batch size
n_steps = 1e5
batch_size = 2000
lr = 0.01

# Model save
vae_model_path = './save/'
model_name = 'model.ckpt'
vae_summary_name = 'vae_train'
snapshot_on = 50

# test
test_iter = 100


#
