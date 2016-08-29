DataName = 'MNIST'
input_dim = 784
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 20
regularization_weight = 0
euclidean_weight = 1.0

# Iteration Number and batch size
n_steps = 1e5
batch_size = 1000



# Model save
vae_model_path = './save/'
model_name = 'model.ckpt'
vae_summary_name = 'vae_train'
snapshot_on = 50


# test
test_iter = 100


NOISE = 0
LEFT_OCC = 1
RIGHT_OCC = 2
TOP_OCC = 3
BOTTOM_OCC = 4
OCC_POPULATION = [NOISE, LEFT_OCC, RIGHT_OCC, TOP_OCC, BOTTOM_OCC]
OCC_PERCENTAGE_MIN = 0.2 # 20%
OCC_PERCENTAGE_MAX = 0.5 # 50%