import fuel.datasets.cifar10 as cifar10
import random as rd
from configuration import *
import numpy as np

from matplotlib import pyplot as plt

train = cifar10.CIFAR10(('train',))
test = cifar10.CIFAR10(('test',))


def sort_dataset_and_get_index(dataset):
    images,labels = dataset.data_sources
    length = len(labels)
    labels = labels.reshape((1, length))
    indexs = labels.argsort()
    dataset.data_sources = (images[indexs, :, :, :].reshape((length, 3, 32,32)), labels[0, indexs].reshape((length, 1)))
    num_images_per_class = int(length/10)
    return {
        'num_images_per_class': num_images_per_class,
        'index':[range(i*num_images_per_class,(i+1)*num_images_per_class) for i in range(10)]
    }

train_label_index = sort_dataset_and_get_index(train)
test_label_index = sort_dataset_and_get_index(test)

# rd.sample(range(train.num_examples), batch_size)
def next_batch(type, batch_size, label=-1):
    images = None
    labels = None
    start = 0
    dataset = None
    index = None

    if type == 'train':
        dataset = train
        index = train_label_index['index']
    elif type == 'test':
        dataset = test
        index = test_label_index['index']

    if label == -1:
        slicing = rd.sample(range(dataset.num_examples), batch_size)
    else:
        slicing = rd.sample(index[label], batch_size)

    images, labels = dataset.get_data(None, slicing)
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



# images, labels = next_batch('train', 100, label=0)