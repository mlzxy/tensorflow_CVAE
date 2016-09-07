from configuration import *
import numpy as np
import random

def randomOcclude(images):
    def _occ(img):
        # plt.imshow(img.reshape((28, 28)))
        OCC_PERCENTAGE = random.uniform(OCC_PERCENTAGE_MIN, OCC_PERCENTAGE_MAX)
        x = random.choice(OCC_POPULATION)
        if x == NOISE:
            img = img.reshape((1, 784))
            img[0, np.random.choice(784, int(OCC_PERCENTAGE * 784))] = 0

        elif x == LEFT_OCC:
            img = img.reshape((28,28))
            img[:, 0:int(OCC_PERCENTAGE * 28)] = 0
            img = img.reshape((1, 784))
        elif x == RIGHT_OCC:
            img = img.reshape((28,28))
            img[:, int((1-OCC_PERCENTAGE) * 28):] = 0
            img = img.reshape((1, 784))
        elif x == TOP_OCC:
            img = img.reshape((28,28))
            img[:int(OCC_PERCENTAGE * 28), :] = 0
            img = img.reshape((1, 784))
        elif x == BOTTOM_OCC:
            img = img.reshape((28,28))
            img[int((1-OCC_PERCENTAGE) * 28):, :] = 0
            img = img.reshape((1, 784))

        # plt.imshow(img.reshape((28, 28)))
        return img

    s1, _ = images.shape
    for i in range(s1):
        images[i, :] = _occ(images[i, :])
    return images
