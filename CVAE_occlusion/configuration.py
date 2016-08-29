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

regularization_weight = 0.01
euclidean_weight = 1.0

# Iteration Number and batch size
n_steps = 10000
batch_size = 1000
lr = 0.02
momentum = 0.6
decay = 0.8
optimizer = Adam


NOISE = 0
LEFT_OCC = 1
RIGHT_OCC = 2
TOP_OCC = 3
BOTTOM_OCC = 4
OCC_POPULATION = [NOISE, LEFT_OCC, RIGHT_OCC, TOP_OCC, BOTTOM_OCC]
OCC_PERCENTAGE_MIN = 0.2 # 20%
OCC_PERCENTAGE_MAX = 0.5 # 50%
# Model save
cvae_model_path = './save/'
model_name = 'model.ckpt'
cvae_summary_name = 'cvae_train'
snapshot_on = 50


LOSS_LIMIT = 50
# test
test_iter = 100