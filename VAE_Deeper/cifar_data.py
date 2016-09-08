import fuel.datasets.cifar10 as cifar10
import random
from configuration import *
import numpy as np

train = cifar10.CIFAR10(('train',))
test = cifar10.CIFAR10(('test',))

def next_batch(type, batch_size):
    images = None
    labels = None
    if type == 'train':
        images, labels = train.get_data(None, random.sample(range(train.num_examples), batch_size))
    elif type == 'test':
        images, labels = test.get_data(None, random.sample(range(test.num_examples), batch_size))

    if CHANNEL == 1:
        images = np.delete(images,  (1,2), axis=1)
    elif CHANNEL == 2:
        images = np.delete(images, (2), axis=1)

    return images, labels



def im2double(images):
    shape = images.shape
    def _im2double(img):
        return img/255

    for i in range(shape[0]):
        for j in range(shape[1]):
            images[i,j,:,:] =  _im2double(images[i,j,:,:])



# images, labels = next_batch('train', 100)
# print(1)