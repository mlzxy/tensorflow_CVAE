import fuel.datasets.cifar10 as cifar10
import random

train = cifar10.CIFAR10(('train',))
test = cifar10.CIFAR10(('test',))

def next_batch(type, batch_size):
    if type == 'train':
        return train.get_data(None, random.sample(range(train.num_examples), batch_size))
    elif type == 'test':
        return test.get_data(None, random.sample(range(test.num_examples), batch_size))



def im2double(images):
    shape = images.shape
    def _im2double(img):
        return img/255

    for i in range(shape[0]):
        for j in range(shape[1]):
            images[i,j,:,:] =  _im2double(images[i,j,:,:])



# images, labels = next_batch('train', 100)
# print(1)